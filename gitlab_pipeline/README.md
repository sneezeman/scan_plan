# GitLab CI / Apptainer build files

These files are the GitLab-side recipe used by the ESRF Apptainer template
to build and publish the `scan_planner` container image (the Python
package itself is named `scan_plan` and is pip-installed inside; the
deployed module / CLI / SIF all use the `scan_planner` convention).

The version here incorporates the **D-Bus machine-id fix** that was
applied in commit `d11ee57` to the in-repo `apptainer/scan_planner.def`.
Without it, PyQt5/QtDBus crashes at container startup because Apptainer
bind-mounts the host's `/etc/machine-id` over the container's, which
can be empty.

## Files

- `scan_planner.def` — Apptainer/Singularity definition file. Used by
  the `.build-apptainer-image` CI job.
- `.gitlab-ci.yml` — pipeline definition. Includes the ESRF
  `apptainer/admin/templates` template and exposes `scan_planner` as
  the runscript entry point.

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
4. `libxcursor1` added to apt install list to silence the
   `Failed to load Xcursor library` VTK warning at runtime.
5. Renamed CLI command and SIF / module artifacts from `scan-plan` /
   `scan_plan` to `scan_planner` to align with the colleague's already-
   deployed module name.

## Deployment

These files live in this directory as a reference. To deploy:

1. Copy `scan_planner.def` to the location expected by the ESRF template
   (typically the repo root, or wherever the template's `BUILD_DEF`
   variable points).
2. Copy `.gitlab-ci.yml` to the repo root of the GitLab project that
   triggers the build.

## Security note

`scan_planner.def` clones the source with GitLab's ephemeral
`CI_JOB_TOKEN`, which exists only for the duration of the build job.
**Never inline a deploy token here** — this repository is public, and a
previously inlined token had to be revoked after being exposed.

Two prerequisites for the job token to work:

1. In `artem1706/scan_plan` → **Settings → CI/CD → Job token permissions**,
   add the project that runs the build (`Apptainer/scan_planner`) to the
   allowlist. Without this the clone gets 403.
2. `.gitlab-ci.yml` must set `APPTAINER_TEMPLATE_VARIABLES: "CI_JOB_TOKEN"`
   on the build job. The ESRF template writes each listed name into a
   `--build-arg-file`, which is what substitutes the token into the
   definition file's build variable.

Because the token arrives as an Apptainer *build variable*, the definition
file must contain the `{{ … }}` braces **exactly once** — in the clone URL.
Apptainer parses every occurrence in the file, including ones inside
comments and `echo` strings, and aborts with
`build var CI_JOB_TOKEN is not defined` if any of them is unsupplied.

The job token is ephemeral: it expires when the build job finishes, so
even if it were captured it cannot be reused.

The `%post` section also deletes `/sources` after `pip install`: a git
checkout keeps the credentialed remote URL in `.git/config`, which would
otherwise ship inside the image published to CVMFS.
