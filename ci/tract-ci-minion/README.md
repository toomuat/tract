# Minions bench infrastructure

## Principles

* build is triggered on each commit, it will run `./.travis/native.sh` to
    perform x86_64 builds, plus a series of `./.travis/cross.sh` for as many
    arm boards configurations.
* `.travis/cross.sh` pushes a `.tgz` to a s3 bucket for each configuration. The
    bundle contains a `entrypoint.sh` script and anything it depends on,
    including the relevant `tract` cli executable. The script is actually names
    `bundle-entrypoint.sh` in the repository.
* devices are running `tract-ci-minion` and will pick the new bundles from the s3 bucket,
    untar and run the `entrypoint.sh`

## Testing locally

```
cargo build --release -p tract && cargo bench -p tract-linalg --no-run && .travis/run-bundle.sh `.travis/make_bundle.sh`
```

## minion agent build

```
cargo install cargo-dinghy
```

Toolchains downloaded from musl.cc.

In ~/dinghy.toml:

```
[platforms.aarch64-linux-musl]
rustc_triple='aarch64-unknown-linux-musl'
toolchain="<path-to>/musl.cc/aarch64-linux-musl-cross"

[platforms.armv7-linux-musl]
rustc_triple='armv7-unknown-linux-musleabihf'
toolchain="<path-to>/musl.cc/armv7l-linux-musleabihf"
env = { TARGET_CFLAGS="-mfpu=neon" }
```

```
cd ci/tract-ci-minion
cargo dinghy -d armv7-linux-musl build --release 
cargo dinghy -d aarch64-linux-musl build --release 
```

## On board setup 

```
MINION=user@hostname.local
scp minion.toml $MINION:.minion.toml
scp target/aarch64-unknown-linux-musl/release/tract-ci-minion $MINION:
scp target/armv7-unknown-linux-musleabihf/release/tract-ci-minion $MINION:
```

On device: edit `.minion.toml`. At this point, running `./tract-ci-minion` should work.

## systemd

in /etc/systemd/system/tract-ci-minion.service. Adjust ExecStart and User.

```
[Unit]
Description=Tract ci minion
After=network-online.target
Wants=network-online.target systemd-networkd-wait-online.service

[Service]
User=ubuntu
ExecStart=/home/ubuntu/tract-ci-minion -v --no-log-prefix -c /home/ubuntu/.minion.toml
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

then

```
systemctl enable tract-ci-minion.timer
systemctl start tract-ci-minion.timer
```

# Setup file server (http only)

```
sudo apt install nginx awscli vim
```

* setup aws credentials (.aws/credentials)
* in $HOME/sync-data.sh:

```

```
* chmod +x $HOME/sync-data.sh
* run it: ./sync-data.sh
* `crontab -e`

```
*/5 * * * * $HOME/sync-data.sh
```


* `sudo vi /etc/nginx/sites-available/models`

```
server {
    root /home/raspbian/models/;

    location /models {
    }
}
```

* `sudo ln -s /etc/nginx/sites-available/models /etc/nginx/sites-enabled/`
* `sudo rm /etc/nginx/sites-enabled/default`
* `sudo /etc/init.d/nginx reload`
* test : `curl -I http://localhost/hey_snips_v1.pb`
