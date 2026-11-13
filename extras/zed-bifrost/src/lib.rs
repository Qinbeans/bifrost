//! Zed extension for the Bifrost programming language.
//!
//! Syntax highlighting, outline, brackets and indentation come from the
//! tree-sitter grammar and the queries in `languages/bifrost/`. This crate only
//! launches the Bifrost language server (`bfc lsp`).
//!
//! The server binary is resolved like the VS Code extension does it:
//!
//! 1. `lsp.bifrost-lsp.binary.path` from Zed settings (with optional
//!    `binary.arguments` and `binary.env`);
//! 2. the `bfc` of a `.venv` at the worktree root, or in one of its immediate
//!    (non-hidden) subfolders;
//! 3. `bfc` on the worktree shell's `PATH`.
//!
//! Extensions run in a WASI sandbox that cannot stat or list the filesystem,
//! and `Worktree::read_text_file` only sees files Zed has scanned. `.venv` is
//! almost always git-ignored (uv even writes a `.venv/.gitignore` of `*`), and
//! Zed does not scan ignored directories, so probing `.venv/bin/bfc` from the
//! extension fails exactly in the common case. On Unix the lookup is therefore
//! done at launch time by a small `/bin/sh` script, which also allows the
//! subfolder search the VS Code extension does. On Windows there is no such
//! shell to rely on, so steps 2 and 3 are: `.venv\Scripts\bfc.exe` if
//! `.venv\pyvenv.cfg` is visible to Zed, else `bfc` on `PATH`, else
//! `.venv\Scripts\bfc.exe` at the root anyway (the spawn error names it).

use zed_extension_api::{self as zed, settings::LspSettings, LanguageServerId, Result};

/// Settings key under `lsp` and the id in `extension.toml`.
const SERVER_ID: &str = "bifrost-lsp";
/// The compiler executable that hosts the language server.
const BINARY: &str = "bfc";
/// Arguments used when `binary.arguments` is not set.
const DEFAULT_ARGS: &[&str] = &["lsp"];

/// `sh -c` script that runs the first `bfc` found, in the VS Code order.
/// `$1` is the worktree root; the remaining arguments are passed to `bfc`.
const UNIX_LAUNCHER: &str = r#"root=$1
shift
for bfc in "$root/.venv/bin/bfc" "$root"/*/.venv/bin/bfc; do
  if [ -f "$bfc" ] && [ -x "$bfc" ]; then
    exec "$bfc" "$@"
  fi
done
if command -v bfc >/dev/null 2>&1; then
  exec bfc "$@"
fi
echo "bifrost: could not find the Bifrost compiler (bfc): no .venv/bin/bfc in $root or its immediate subfolders, and bfc is not on PATH. Install it, or set lsp.bifrost-lsp.binary.path in your Zed settings." >&2
exit 127
"#;

struct BifrostExtension;

impl BifrostExtension {
    /// The command and leading arguments that start `bfc` without a
    /// configured path.
    fn discovered_command(worktree: &zed::Worktree) -> (String, Vec<String>) {
        let root = worktree.root_path();
        let root = root.trim_end_matches(['/', '\\']);
        let (os, _) = zed::current_platform();
        match os {
            zed::Os::Mac | zed::Os::Linux => (
                "/bin/sh".to_string(),
                vec![
                    "-c".to_string(),
                    UNIX_LAUNCHER.to_string(),
                    "bifrost-lsp".to_string(),
                    root.to_string(),
                ],
            ),
            zed::Os::Windows => {
                let venv = format!("{root}\\.venv\\Scripts\\{BINARY}.exe");
                let command = if worktree.read_text_file(".venv/pyvenv.cfg").is_ok() {
                    venv
                } else {
                    worktree.which(BINARY).unwrap_or(venv)
                };
                (command, Vec::new())
            }
        }
    }
}

impl zed::Extension for BifrostExtension {
    fn new() -> Self {
        BifrostExtension
    }

    fn language_server_command(
        &mut self,
        _language_server_id: &LanguageServerId,
        worktree: &zed::Worktree,
    ) -> Result<zed::Command> {
        let binary = LspSettings::for_worktree(SERVER_ID, worktree)
            .ok()
            .and_then(|settings| settings.binary);
        let (path, arguments, extra_env) = match binary {
            Some(binary) => (binary.path, binary.arguments, binary.env),
            None => (None, None, None),
        };
        let arguments =
            arguments.unwrap_or_else(|| DEFAULT_ARGS.iter().map(ToString::to_string).collect());

        let (command, mut args) = match path.filter(|path| !path.trim().is_empty()) {
            Some(path) => (path, Vec::new()),
            None => Self::discovered_command(worktree),
        };
        args.extend(arguments);

        let mut env = worktree.shell_env();
        if let Some(extra_env) = extra_env {
            for (key, value) in extra_env {
                env.retain(|(existing, _)| existing != &key);
                env.push((key, value));
            }
        }

        Ok(zed::Command { command, args, env })
    }

    fn language_server_initialization_options(
        &mut self,
        _language_server_id: &LanguageServerId,
        worktree: &zed::Worktree,
    ) -> Result<Option<zed::serde_json::Value>> {
        Ok(LspSettings::for_worktree(SERVER_ID, worktree)
            .ok()
            .and_then(|settings| settings.initialization_options))
    }

    fn language_server_workspace_configuration(
        &mut self,
        _language_server_id: &LanguageServerId,
        worktree: &zed::Worktree,
    ) -> Result<Option<zed::serde_json::Value>> {
        Ok(LspSettings::for_worktree(SERVER_ID, worktree)
            .ok()
            .and_then(|settings| settings.settings))
    }
}

zed::register_extension!(BifrostExtension);
