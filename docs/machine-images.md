# Machine images

Machine images are benchmark inputs, not package data. Keep them outside the
repository, pass their paths explicitly, and put disposable overlays on local
scratch. OSWorld hashes the complete selected image before evaluation and stores
that identity in the immutable run manifest. If `IMAGE.provenance.json` exists,
it must be a regular non-symlink JSON file whose top-level
`final_image_sha256` matches `IMAGE`.

## OSWorld v1

The pinned OSWorld v1 manager names the official Ubuntu archive. Resolve it at
the observed immutable dataset revision rather than through mutable `main`:

```bash
curl --fail --location --continue-at - \
  https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/9600484566f238a9ce57ea32c33567c6044e41d8/Ubuntu.qcow2.zip \
  --output /assets/osworld/Ubuntu.qcow2.zip
echo 'b795b6cd4c69b252c1b4f10150a347795555032501b60fd031751ed09b896712  /assets/osworld/Ubuntu.qcow2.zip' \
  | sha256sum --check -
unzip /assets/osworld/Ubuntu.qcow2.zip -d /assets/osworld
qemu-img check /assets/osworld/Ubuntu.qcow2
```

The validated extracted image is 24,460,197,888 bytes, has a 50 GiB virtual
size, and SHA-256
`6bf667a852b3c307f61d9f09c42559351f45e0607e428b4997becf534cf4d313`.

Docker-less hosts can materialize the exact official container manifest as an
Apptainer SIF:

```bash
apptainer pull /assets/osworld/osworld-docker.sif \
  docker://happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9
```

SIF bytes include local build metadata, so the OCI manifest digest is the
portable source pin and each run records the observed full SIF hash. The
integrated compatibility client invokes the image's official entrypoint and
QEMU/disk/UEFI scripts. It replaces only the Docker networking/display layer:
QEMU user-mode `hostfwd` replaces the privileged NET_ADMIN tap, and host noVNC
is disabled. Docker root can open the image's copied mode-`0444` UEFI firmware,
whereas fakeroot QEMU cannot; the adapter therefore extracts the exact firmware
bytes from this SIF into each private launch directory at mode `0600`. A PID
namespace keeps entrypoint daemons inside the launcher lifecycle. The official
entrypoint, guest, OSWorld controller, task reset, actions, and hidden evaluator
remain upstream. These are recorded runtime adaptations, not a claim of
byte-for-byte Docker execution.

## OSWorld v2

The `v2026.06.24` release manifest pins the VM archive tag, path, size, and
archive hash. The task-class dataset is separately gated; obtaining the image
does not grant task access.

```bash
curl --fail --location --continue-at - \
  https://huggingface.co/datasets/xlangai/v2-image/resolve/v2026.06.24/osworld-v2-ubuntu-x86.qcow2.zip \
  --output /assets/osworld-v2/osworld-v2-ubuntu-x86.qcow2.zip
echo 'eb737ae70b49849e24af407de6a518439a23de05a8497096a948334ce0a909aa  /assets/osworld-v2/osworld-v2-ubuntu-x86.qcow2.zip' \
  | sha256sum --check -
unzip /assets/osworld-v2/osworld-v2-ubuntu-x86.qcow2.zip \
  -d /assets/osworld-v2
qemu-img check /assets/osworld-v2/osworld-v2-ubuntu-x86.qcow2
```

The validated archive is 14,189,763,267 bytes. Its extracted image is
27,402,633,216 bytes, has a 50 GiB virtual size, and SHA-256
`3d632f031459583cf936e0c4c5bb939122df0fec85aecb0d044ef2d3e5863335`.
The same content-addressed OSWorld runtime container shown above booted this
image through the pinned v2 checkout and returned a valid 1920×1080 controller
screenshot. This establishes the machine/controller path only: no v2 task or
score is claimed without the gated `xlangai/osworld_v2_tasks` release data.

## Validation-node storage boundary

(The "validation node" is the shared-cluster host the recorded hashes
were produced on; the layout below is a record of that machine, not a
required layout for yours.)

On the 2026-08-12 validation node, both extracted qcow2 files are complete,
match the hashes above, and currently pass `qemu-img check`; the OSWorld SIF also
matches its recorded hash and source OCI manifest. The extracted qcow2 files are
on local NVMe under `/tmp/mini-agent/assets`, not durable storage. The requested
user-owned durable NFS area had only 3.3 GiB free while the two
images require about 48 GiB, so only provenance sidecars and the 81 MB OSWorld
SIF fit there. Do not treat those sidecars as an image backup. Before relying on
this node after scratch cleanup, either allocate durable capacity and copy the
full files with their sidecars or rerun the hash-pinned acquisition commands.
