# Connect to PSNM

## Configure ssh-agent

Be sure that you have followed these instructions : https://www.ens-lyon.fr/PSMN/Documentation/connection/ssh_keys_and_agent.html

## Edit your ssh configuration

You may add these lines in your  `~/.ssh/config` file (don't forget to replace the information between the brackets)

```
Host *
  ServerAliveInterval 60
  ForwardX11Timeout 1d
  TCPKeepAlive yes
  ForwardAgent yes
  ForwardX11 yes         # for Linux
#  ForwardX11Trusted yes # for MacOSX
  Compression yes


HostName ssh.psmn.ens-lyon.fr
  User {your-user}
  IdentitiesOnly yes
  IdentityFile {path-to-where-your-key-is}

Host PSMN_Internal
  HostName allo-psmn.psmn.ens-lyon.fr
  User {your-user}
  IdentitiesOnly yes
  IdentityFile {path-to-where-your-key-is-idem-to-last-path}
  ProxyJump ENS_Lyon

Host PSMN_sr650node230
  HostName sr650node230
  User {your-user}
  IdentitiesOnly yes
  IdentityFile {path-to-where-your-key-is-idem-to-last-path}
  ProxyJump PSMN_Internal
```

## Connect to the visualisation node

```bash
ssh-add # star the ssh agent
Host PSMN_sr650node230
```

## Setup the environnement

You can then setup your python environnement following the instruction in [CBP.md].
Then clone this repository in your HOME folder.


## Connect python notebook

Either connect to PSMN_sr650node230 with vs-code 

1. Launch a notebook server on the visualisation node `jupyter notebook --ip=0.0.0.0 --no-browser
2. Write the port number on which jupyter is running and the token
3. Enable port forwarding with `ssh -L -f localhost:{port-number}:localhost:{port-number} PSMN_sr650node230 -N`
4. Connect to `localhost:{port-number}?token={TOKEN}` from your local browser
