# vscode

Visual Studio Code - Server.

## Usage

### Starting

```bash
make up
```

Access at http://localhost:11843
Default password: `password`

### Git access

1. Add your ssh key to `/config/.ssh/`
2. Open a terminal in VSCode and add your Git user:
   ```
   $ git config --global user.name "username"
   $ git config --global user.email "email address"
   ```

## Troubleshooting

 - If you have trouble installing extensions, go find them online (ex. https://open-vsx.org) and download the latest `.vsx`.
   Put it in a dir vscode has access to, then go to extensions tab, click the `...` at top right, go to install extension via `.vsx`.
