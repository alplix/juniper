BRP4 Binary Radio Pulsar Search - Juniper Metal port (Apple Silicon)
=====================================================================
Port: "Juniper (BRP4 Metal port) by Alperen Yavuz" (github.com/alplix/juniper)
Version: v0.1-dev

What is this?
-------------
An unofficial, from-scratch Metal port of Einstein@Home's BRP4 Binary
Radio Pulsar Search GPU application, for Apple Silicon (M-series) Macs.
Resampling, FFT (via MPSGraph), power spectrum and harmonic summing all
run natively on the GPU -- no CPU fallback in the search pipeline.

Contents of this package
-------------------------
  einsteinbinary_BRP4_macos   the application, single file
  app_info.xml                BOINC anonymous platform
  app_config.xml               BOINC project-folder settings

Requirements
------------
  - macOS on Apple Silicon (M1 or newer)
  - A BOINC client recent enough to report an "apple_gpu" coprocessor
    (lib/coproc.h's PROC_TYPE_APPLE_GPU) -- older BOINC client versions
    won't detect the GPU resource this app_info.xml requests.

Installation (BOINC anonymous platform)
----------------------------------------
1) Open your BOINC project folder:
      ~/Library/Application Support/BOINC Data/projects/einstein.phys.uwm.edu/
   (or wherever your BOINC client's data directory is configured)
2) Back up and DELETE any existing app_info.xml there.
3) Copy ALL files from this package into that folder and make sure the
   binary is executable:
      chmod +x einsteinbinary_BRP4_macos
4) Restart BOINC and make sure "Use GPU" is enabled in the client settings.
The declared version number is deliberately 99: project guidance asks custom
builds to stay below 100 so they never collide with official versions.

Standalone usage (no BOINC needed for testing)
------------------------------------------------
  ./einsteinbinary_BRP4_macos \
      -i input.binary -t bank.txt -l zaplist.txt \
      -o results.dat -c checkpoint.dat -W -D 0 -z

Key flags: -i input time series, -t template bank, -l zap list,
-o candidate output, -c checkpoint file (delete it before a fresh run).
Note: run-time log output (device selection, progress, candidate
statistics) is written to stderr.txt in the working directory once BOINC's
diagnostics initialize, not to the process's own stdout/stderr.

Known limitation (IMPORTANT -- READ BEFORE ENABLING LIVE BOINC WORK)
-----------------------------------------------------------------------
The CURRENT Einstein@Home Ter5 "sband_dns" tasks belong to the
new-generation official application (BRP7, MeerKAT data), whose new
command-line options (--pb_min, --start_template_id, ...) and source code
are not publicly released. This build -- like the companion CUDA port it's
built alongside -- only understands classic "-t bank"-style work. If you
advertise this app_info.xml on a live BOINC connection, the Einstein@Home
scheduler may send you current-generation Ter5/BRP7 work, which THIS BUILD
CANNOT PROCESS ("unrecognized option --pb_min", exit 4) -- every such task
will fail and may affect your host's reported reliability.
Safe usage right now: standalone tests with your own or synthetic
(test/make_synthetic.c) input. Do not enable this on a live project
connection expecting current Einstein@Home Ter5 work to succeed.

Validation status (honest)
---------------------------
Candidate output diffed against the companion CUDA port (brp4-cuda-port)
running identical synthetic input: zero frequency/orbital-template/
harmonic-count mismatches across both a strong- and weak-signal synthetic
work unit, power values agreeing to floating-point rounding level between
two independent FFT implementations. NOT yet run against a real
Einstein@Home work unit of any kind -- see the project README for the full
validation writeup and performance numbers (github.com/alplix/juniper).

License
-------
The upstream Einstein@Home BRP4 source is GPL v2+; this build is
distributed under the same license.
