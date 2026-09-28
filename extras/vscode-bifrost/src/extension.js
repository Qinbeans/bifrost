// Starts the Bifrost language server (`bifrost lsp`) for .bif files.
const fs = require("fs");
const path = require("path");
const vscode = require("vscode");
const { LanguageClient } = require("vscode-languageclient/node");

let client;

/** The `bifrost` executable of a project virtualenv under `folder`, if any. */
function venvExecutable(folder) {
  const candidates = [
    path.join(folder, ".venv", "bin", "bifrost"),
    path.join(folder, ".venv", "Scripts", "bifrost.exe"),
  ];
  return candidates.find((candidate) => fs.existsSync(candidate));
}

/**
 * The configured `bifrost.server.path`; else a `.venv` in a workspace folder
 * or one of its immediate subfolders; else `bifrost` on PATH.
 */
function serverExecutable() {
  const configured = vscode.workspace.getConfiguration("bifrost").get("server.path");
  if (configured) {
    return configured;
  }
  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    const root = folder.uri.fsPath;
    const found = venvExecutable(root);
    if (found) {
      return found;
    }
    let entries = [];
    try {
      entries = fs.readdirSync(root, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      if (entry.isDirectory() && !entry.name.startsWith(".")) {
        const nested = venvExecutable(path.join(root, entry.name));
        if (nested) {
          return nested;
        }
      }
    }
  }
  return "bifrost";
}

async function activate(context) {
  const command = serverExecutable();
  client = new LanguageClient(
    "bifrost",
    "Bifrost",
    { command, args: ["lsp"] },
    {
      documentSelector: [{ scheme: "file", language: "bifrost" }],
      synchronize: {
        fileEvents: vscode.workspace.createFileSystemWatcher("**/config.yaml"),
      },
    },
  );
  context.subscriptions.push(
    vscode.commands.registerCommand("bifrost.restartServer", () => client.restart()),
  );
  try {
    await client.start();
  } catch (error) {
    vscode.window.showErrorMessage(
      `Could not start the Bifrost language server (${command} lsp): ${error.message}. ` +
        "Set bifrost.server.path to your bifrost executable.",
    );
  }
}

async function deactivate() {
  if (client) {
    await client.stop();
  }
}

module.exports = { activate, deactivate };
