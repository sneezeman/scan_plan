# GitLab CI / Apptainer build files

These files are the GitLab-side recipe used by the ESRF Apptainer template
to build and publish the `scan_plan` container image. They were originally
authored by William Chevremont and adapted from the colleague's working
copy.

The version here incorporates the **D-Bus machine-id fix** that was applied
in commit `d11ee57` to the in-repo `apptainer/scan_plan.def`. Without it,
PyQt5/QtDBus crashes at container startup because Apptainer bind-mounts the
host's `/etc/machine-id` over the container's, which can be empty.

## Files

- `scan_plan.def` — Apptainer/Singularity definition file. Used by the
  `.build-apptainer-image` CI job.
- `.gitlab-ci.yml` — pipeline definition. Includes the ESRF
  `apptainer/admin/templates` template and exposes `scan-plan` as the
  runscript entry point.

## Differences vs. the colleague's working copy

1. New `%environment` block sets `DBUS_SESSION_BUS_ADDRESS=disabled:`,
   `QT_QPA_PLATFORM=xcb`, `PYVISTA_OFF_SCREEN=false`.
2. End of `%post` generates a machine-id at build time
   (`/etc/machine-id` + `/var/lib/dbus/machine-id`) so the container has
   a valid one baked in even before the env-var fallback kicks in.
3. `apt` is configured to retry transient failures
   (`/etc/apt/apt.conf.d/99retries`) and `apt-get update` itself is
   wrapped in a 5-attempt retry loop. This is needed because the ESRF CI
   has hit 502/503 errors from `archive.ubuntu.com` /
   `security.ubuntu.com` mid-build, which left the package index
   incomplete and caused `Package 'ca-certificates' has no installation
   candidate` failures.

## Filename note

The CI deploy repo (`apptainer/scan_planner`) uses the filename
`scan_planner.def`, while this reference copy is named `scan_plan.def`
to match the Python project's own naming. Rename when copying into the
CI deploy repo.

## Deployment

These files live in this directory as a reference. To deploy:

1. Copy `scan_plan.def` to the location expected by the ESRF template
   (typically the repo root or wherever the template's `BUILD_DEF` variable
   points).
2. Copy `.gitlab-ci.yml` to the repo root of the GitLab project that
   triggers the build.

## Security note

The deploy token is currently inlined in `scan_plan.def`. Prefer moving it
to a GitLab CI/CD variable (e.g. `CI_DEPLOY_USER`/`CI_DEPLOY_PASSWORD`) and
referencing it via `${VAR}` in the clone URL.
