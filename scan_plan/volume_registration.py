"""
volume_registration.py
Logic for coordinate transformation between Prescan (Volume) and Refscan (Machine).
"""

import pint
import typing
import numpy as np
from scipy.spatial.transform import RigidTransform as Tf
from scipy.spatial.transform import Rotation as R
import scipy.optimize as opt

ureg = pint.UnitRegistry()

class ReferenceVolume(typing.NamedTuple):
    su: float
    sv: float
    sz: float 
    pixel_size: float
    pixel_unit: str
    width: int 
    height: int 
    
class MatchPoint(typing.NamedTuple):
    prescan_coordinate: tuple
    refscan_coordinate: tuple

class RegistrationResult(typing.NamedTuple):
    distances: np.ndarray
    transformed_coords: np.ndarray
    solution: typing.Any
    rotation_angles: list # [yaw, pitch, roll]

class VolumeRegistration():
    
    def __init__(self, prescan_pixelsize, prescan_pixelunits='nm', flipped=False, pitch=-0.015396, optics=None):
        if optics is None:
            optics = {}
        self.__prescan_pixelsize = prescan_pixelsize
        self.__prescan_pixelunits = prescan_pixelunits
        self.__ref_volumes = []
        self.__datapoints = []
        self.__transformationMatrix = None
        self.__beam_pitch = optics.get('beam_pitch_rad', pitch)
        self.__optics_pixel_size = optics.get('optics_pixel_size_um', 2.952)
        self.__z12 = optics.get('z12', 1281)
        self.__sx0 = optics.get('sx0_mm', 1.28)
        self._rotation_offset = optics.get('rotation_offset_deg', -21.5)
        self._prescan_offset_scaled = np.array([0,0,0])
        self._refscan_offset = np.array([0,0,0])
        
    def addReferenceVolume(self, su, sv, sz, pixel_size, pixel_unit='nm', width=2048, height=2048):
        idx = len(self.__ref_volumes)
        self.__ref_volumes += [ ReferenceVolume(su, sv, sz, pixel_size, pixel_unit, width, height) ]
        return idx
    
    def _suv2saxy(self, su, sv):
        th = -self._rotation_offset*np.pi/180.
        sax = -su*np.sin(th) + sv*np.cos(th)
        say =  su*np.cos(th) + sv*np.sin(th)
        return sax, say

    def _saxy2suv(self, sax, say):
        th = -self._rotation_offset*np.pi/180.
        su = say*np.cos(th) - sax*np.sin(th)
        sv = say*np.sin(th) + sax*np.cos(th)
        return su, sv
        
    def addMatchPoint(self, prescan_coordinate: tuple, refscan_coordinate: tuple, ref_id : int):
        assert isinstance(prescan_coordinate, tuple)
        assert isinstance(refscan_coordinate, tuple)
        
        if len(self.__ref_volumes) == 0 : raise RuntimeError("At least one reference volume must be given!")
        if ref_id >= len(self.__ref_volumes): raise RuntimeError("Please insert your reference volume first!")
            
        if ref_id > 0:
            init_coordinate = refscan_coordinate
            refvol  = self.__ref_volumes[ref_id]
            ref0vol = self.__ref_volumes[0]
            
            sax0, say0 = self._suv2saxy(ref0vol.su, ref0vol.sv)
            sax, say = self._suv2saxy(refvol.su, refvol.sv)
            
            ps = ureg.Quantity(refvol.pixel_size, refvol.pixel_unit).to('mm').magnitude
            ps0 = ureg.Quantity(ref0vol.pixel_size, ref0vol.pixel_unit).to('mm').magnitude
            
            dx = (init_coordinate[0] - refvol.width/2.) * ps
            dy = (init_coordinate[1] - refvol.width/2.) * ps
            dz = (init_coordinate[2] - refvol.height/2.) * ps
            
            X = (sax + dx - sax0)/ps0
            Y = (say + dy - say0)/ps0
            Z = (refvol.sz + dz - ref0vol.sz)/ps0
            refscan_coordinate = (X,Y,Z)
            
        self.__datapoints += [ MatchPoint(prescan_coordinate, refscan_coordinate), ]
        
    def _getTransformation(self, yaw, pitch, roll, tx, ty, tz):
        t_A_B = np.array([tx, ty, tz])
        r_A_B = R.from_euler('xyz',[yaw, pitch, roll], degrees=True)        
        Tf_A_B = Tf.from_components(t_A_B, r_A_B)
        return Tf_A_B
        
    def fitTransformationMatrix(self, rot_z_only=False, method='optimizer'):
        """
        Calculates transformation using either SciPy Optimizer or SVD Kabsch algorithm.
        """
        ndata = len(self.__datapoints)
        refscan_coords = np.empty((ndata, 3))
        prescan_coords = np.empty((ndata, 3))
        
        for i in range(ndata):
            refscan_coords[i] = self.__datapoints[i].refscan_coordinate
            prescan_coords[i] = self.__datapoints[i].prescan_coordinate
            
        prescan_coords_scaled = self._scale_prescan(prescan_coords)
        self._prescan_offset_scaled = np.mean(prescan_coords_scaled, axis=0)
        self._refscan_offset = np.mean(refscan_coords, axis=0)

        P = prescan_coords_scaled - self._prescan_offset_scaled
        Q = refscan_coords - self._refscan_offset

        if method == 'svd':
            # --- SVD / Kabsch Algorithm ---
            if rot_z_only:
                H = np.dot(P[:, :2].T, Q[:, :2])
                U, S, Vt = np.linalg.svd(H)
                R_2d = np.dot(Vt.T, U.T)
                
                # Check reflection
                if np.linalg.det(R_2d) < 0:
                    Vt[-1, :] *= -1
                    R_2d = np.dot(Vt.T, U.T)
                    
                R_mat = np.eye(3)
                R_mat[:2, :2] = R_2d
            else:
                H = np.dot(P.T, Q)
                U, S, Vt = np.linalg.svd(H)
                R_mat = np.dot(Vt.T, U.T)
                
                if np.linalg.det(R_mat) < 0:
                    Vt[-1, :] *= -1
                    R_mat = np.dot(Vt.T, U.T)

            # Convert Rotation matrix to Euler angles
            final_angles = R.from_matrix(R_mat).as_euler('xyz', degrees=True)
            self.__transformationMatrix = self._getTransformation(*final_angles, 0, 0, 0)
            
            # Create a dummy solution object for compatibility
            class SVDSolution: pass
            sol = SVDSolution()
            sol.fun = np.sum(np.linalg.norm(Q - self.__transformationMatrix.apply(P), axis=1))
            sol.message = "SVD (Kabsch) Exact Mathematical Solution"
            
        else:
            # --- SciPy Optimizer ---
            def getTransformedCoordinates(coords, yaw, pitch, roll, tx, ty, tz):
                Tf_A_B = self._getTransformation(yaw, pitch, roll, tx, ty, tz)
                return Tf_A_B.apply(coords)
            
            def getQuality(X, refscan_coords, prescan_coords):
                tx, ty, tz = 0, 0, 0
                yaw, pitch, roll = X
                transformed = getTransformedCoordinates(prescan_coords, yaw, pitch, roll, tx, ty, tz)
                return np.sum(np.sum((refscan_coords - transformed)**2, axis=1)**0.5)
            
            def getQualityYaw(X, refscan_coords, prescan_coords):
                tx, ty, tz = 0, 0, 0
                yaw = 0
                pitch = 0
                roll = X[0]
                transformed = getTransformedCoordinates(prescan_coords, yaw, pitch, roll, tx, ty, tz)
                return np.sum(np.sum((refscan_coords - transformed)**2, axis=1)**0.5)

            if rot_z_only:
                x0 = [0,]
                steps = [90,]
                initial_simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))])
                sol = opt.minimize(getQualityYaw, x0, (Q, P), 'Nelder-Mead', options={'initial_simplex': initial_simplex})
                self.__transformationMatrix = self._getTransformation(0, 0, sol.x[0], 0, 0, 0)
                final_angles = [0, 0, sol.x[0]]
            else:
                x0 = [0,0,0]
                steps = [90,90,90]
                initial_simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))])
                sol = opt.minimize(getQuality, x0, (Q, P), 'Nelder-Mead', options={'initial_simplex': initial_simplex})
                self.__transformationMatrix = self._getTransformation(*sol.x, 0, 0, 0)
                final_angles = list(sol.x)
        
        transformed = self.transformToRefscan(prescan_coords)
        distances = np.sum((refscan_coords - transformed)**2, axis=1)**0.5
        
        return RegistrationResult(distances, transformed, sol, final_angles)

    def _scale_prescan(self, prescan_coords):
        return prescan_coords * (ureg.Quantity(self.__prescan_pixelsize, self.__prescan_pixelunits) / ureg.Quantity(self.__ref_volumes[0].pixel_size, self.__ref_volumes[0].pixel_unit)).to('dimensionless').magnitude
            
    def transformToRefscan(self, prescan_coords):
        prescan_coords_scaled = self._scale_prescan(prescan_coords)
        return self.__transformationMatrix.apply(prescan_coords_scaled - self._prescan_offset_scaled) + self._refscan_offset
    
    def refscan_to_motors(self, refscan_coords, scan_pixel_size, scan_pixel_unit='nm'):
        ref0vol = self.__ref_volumes[0]
        deltas = (refscan_coords - np.array([ref0vol.width, ref0vol.width, ref0vol.height], dtype=np.float32)/2.) * ureg.Quantity(ref0vol.pixel_size, ref0vol.pixel_unit).to("mm")
        
        sax_deltas = -deltas[:,1] 
        say_deltas = -deltas[:,0] 
        sz_deltas  =  deltas[:,2]
        
        su_deltas, sv_deltas = self._saxy2suv(sax_deltas, say_deltas)
        
        final_su = su_deltas.to("mm").magnitude + ref0vol.su
        final_sv = sv_deltas.to("mm").magnitude + ref0vol.sv
        sz_d1 = sz_deltas.to("mm").magnitude + ref0vol.sz
        
        M0 = ((ureg.Quantity(self.__optics_pixel_size, 'um') / ureg.Quantity(ref0vol.pixel_size, ref0vol.pixel_unit))).to('dimensionless').magnitude
        z1r = self.__z12/M0
        sxr = z1r + self.__sx0
        
        M = ((ureg.Quantity(self.__optics_pixel_size, 'um') / ureg.Quantity(scan_pixel_size, scan_pixel_unit))).to('dimensionless').magnitude
        z1s = self.__z12/M
        sxd1= z1s + self.__sx0

        sz_d1 = sz_d1 + (sxd1 - sxr)*np.sin(self.__beam_pitch)
        return final_su, final_sv, sz_d1

    def motors_to_refscan(self, su_coords, sv_coords, sz_coords, scan_pixel_size, scan_pixel_unit='nm'):
        """Convert motor coordinates (su, sv, sz in mm) back to refscan pixel coordinates.

        This is the inverse of refscan_to_motors(). Useful when the user identifies
        features directly on the physical sample by moving the stage and recording
        su/sv/sz motor positions, without performing a refscan.
        """
        ref0vol = self.__ref_volumes[0]
        ps = ureg.Quantity(ref0vol.pixel_size, ref0vol.pixel_unit).to("mm").magnitude

        # Reverse the beam-pitch correction on sz
        M0 = (ureg.Quantity(self.__optics_pixel_size, 'um') / ureg.Quantity(ref0vol.pixel_size, ref0vol.pixel_unit)).to('dimensionless').magnitude
        z1r = self.__z12 / M0
        sxr = z1r + self.__sx0
        M = (ureg.Quantity(self.__optics_pixel_size, 'um') / ureg.Quantity(scan_pixel_size, scan_pixel_unit)).to('dimensionless').magnitude
        z1s = self.__z12 / M
        sxd1 = z1s + self.__sx0
        sz_corrected = sz_coords - (sxd1 - sxr) * np.sin(self.__beam_pitch)

        # Motor deltas from reference volume origin (mm)
        su_deltas = su_coords - ref0vol.su
        sv_deltas = sv_coords - ref0vol.sv
        sz_deltas = sz_corrected - ref0vol.sz

        # Inverse of _saxy2suv: convert su/sv deltas back to sax/say
        sax_deltas, say_deltas = self._suv2saxy(
            ureg.Quantity(su_deltas, 'mm'),
            ureg.Quantity(sv_deltas, 'mm')
        )

        # Inverse of the sign/axis swap in refscan_to_motors:
        #   Forward: sax_deltas = -deltas[:,1], say_deltas = -deltas[:,0], sz_deltas = deltas[:,2]
        #   Inverse: deltas[:,0] = -say_deltas, deltas[:,1] = -sax_deltas, deltas[:,2] = sz_deltas
        refscan_x = -say_deltas.to("mm").magnitude / ps + ref0vol.width / 2.0
        refscan_y = -sax_deltas.to("mm").magnitude / ps + ref0vol.width / 2.0
        refscan_z = sz_deltas / ps + ref0vol.height / 2.0

        return np.column_stack([refscan_x, refscan_y, refscan_z])