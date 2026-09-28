"""A language server for Bifrost, speaking LSP over stdio (``bifrost lsp``).

It reports syntax, lowering and type errors as you type, formats documents,
outlines their top-level bindings, jumps to definitions (externs jump to their
``config.yaml`` declaration), shows signatures on hover, and completes names
and ``module.`` members.
"""

import functools
import traceback
from collections.abc import Callable
from pathlib import Path

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from bifrost.formatter import FormatError, format_source
from bifrost.server.analysis import Document, Range, Severity, SymbolKind

server = LanguageServer("bifrost", "0.1.0")

_documents: dict[str, Document] = {}

_SEVERITIES = {
    Severity.ERROR: types.DiagnosticSeverity.Error,
    Severity.WARNING: types.DiagnosticSeverity.Warning,
    Severity.INFORMATION: types.DiagnosticSeverity.Information,
    Severity.HINT: types.DiagnosticSeverity.Hint,
}

_SYMBOL_KINDS = {
    SymbolKind.FUNCTION: types.SymbolKind.Function,
    SymbolKind.STRUCT: types.SymbolKind.Struct,
    SymbolKind.MODULE: types.SymbolKind.Module,
    SymbolKind.CONSTANT: types.SymbolKind.Constant,
}

_COMPLETION_KINDS = {
    "function": types.CompletionItemKind.Function,
    "struct": types.CompletionItemKind.Struct,
    "module": types.CompletionItemKind.Module,
    "constant": types.CompletionItemKind.Constant,
    "variable": types.CompletionItemKind.Variable,
    "keyword": types.CompletionItemKind.Keyword,
    "type": types.CompletionItemKind.TypeParameter,
}


def _safely[**P, R](handler: Callable[P, R]) -> Callable[P, R | None]:
    """Log a failing handler's error instead of failing the request.

    Analysis runs on half-typed code, so a bug on some unusual tree should cost
    one hover or outline, not an error in the editor on every keystroke.
    """

    @functools.wraps(handler)
    def run(*args: P.args, **kwargs: P.kwargs) -> R | None:
        try:
            return handler(*args, **kwargs)
        except Exception:  # noqa: BLE001 - reported to the client's log instead
            server.window_log_message(
                types.LogMessageParams(type=types.MessageType.Error, message=traceback.format_exc())
            )
            return None

    return run


def _range(span: Range) -> types.Range:
    (start_line, start_character), (end_line, end_character) = span
    return types.Range(
        types.Position(line=start_line, character=start_character),
        types.Position(line=end_line, character=end_character),
    )


def _document(uri: str) -> Document:
    """Return the analysis of ``uri``'s current text, reparsed when the text changes."""
    text_document = server.workspace.get_text_document(uri)
    cached = _documents.get(uri)
    if cached is None or cached.text != text_document.source:
        cached = Document.open(Path(text_document.path), text_document.source)
        _documents[uri] = cached
    return cached


def _publish(uri: str) -> None:
    diagnostics = [
        types.Diagnostic(
            range=_range(d.range),
            message=d.message,
            severity=_SEVERITIES[d.severity],
            source="bifrost",
            tags=[types.DiagnosticTag.Unnecessary] if d.unnecessary else None,
        )
        for d in _document(uri).diagnostics()
    ]
    server.text_document_publish_diagnostics(types.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics))


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
@_safely
def _did_open(params: types.DidOpenTextDocumentParams) -> None:
    _publish(params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
@_safely
def _did_change(params: types.DidChangeTextDocumentParams) -> None:
    _publish(params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
@_safely
def _did_save(params: types.DidSaveTextDocumentParams) -> None:
    _documents.pop(params.text_document.uri, None)  # config.yaml may have changed
    _publish(params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_CLOSE)
@_safely
def _did_close(params: types.DidCloseTextDocumentParams) -> None:
    _documents.pop(params.text_document.uri, None)
    server.text_document_publish_diagnostics(
        types.PublishDiagnosticsParams(uri=params.text_document.uri, diagnostics=[])
    )


@server.feature(types.TEXT_DOCUMENT_FORMATTING)
@_safely
def _formatting(params: types.DocumentFormattingParams) -> list[types.TextEdit] | None:
    document = _document(params.text_document.uri)
    try:
        formatted = format_source(document.text)
    except FormatError:
        return None  # nothing to do until the syntax errors are fixed
    if formatted == document.text:
        return []
    lines = document.text.split("\n")
    end = types.Position(line=len(lines), character=0)
    return [types.TextEdit(range=types.Range(types.Position(line=0, character=0), end), new_text=formatted)]


@server.feature(types.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
@_safely
def _symbols(params: types.DocumentSymbolParams) -> list[types.DocumentSymbol]:
    return [
        types.DocumentSymbol(
            name=symbol.name,
            kind=_SYMBOL_KINDS[symbol.kind],
            detail=symbol.detail,
            range=_range(symbol.range),
            selection_range=_range(symbol.selection),
        )
        for symbol in _document(params.text_document.uri).symbols()
    ]


@server.feature(types.TEXT_DOCUMENT_DEFINITION)
@_safely
def _definition(params: types.DefinitionParams) -> types.Location | None:
    position = (params.position.line, params.position.character)
    location = _document(params.text_document.uri).definition(position)
    if location is None:
        return None
    return types.Location(uri=location.path.as_uri(), range=_range(location.range))


@server.feature(types.TEXT_DOCUMENT_HOVER)
@_safely
def _hover(params: types.HoverParams) -> types.Hover | None:
    position = (params.position.line, params.position.character)
    text = _document(params.text_document.uri).hover(position)
    if text is None:
        return None
    return types.Hover(contents=types.MarkupContent(kind=types.MarkupKind.Markdown, value=text))


@server.feature(types.TEXT_DOCUMENT_INLAY_HINT)
@_safely
def _inlay_hints(params: types.InlayHintParams) -> list[types.InlayHint]:
    return [
        types.InlayHint(
            position=types.Position(line=hint.position[0], character=hint.position[1]),
            label=hint.label,
            kind=types.InlayHintKind.Parameter if hint.parameter else types.InlayHintKind.Type,
            padding_right=hint.parameter,  # `status: 200`, not `status:200`
        )
        for hint in _document(params.text_document.uri).inlay_hints()
    ]


# `.` for members, `"` and `:` for module names in `import("std:...")`.
@server.feature(types.TEXT_DOCUMENT_COMPLETION, types.CompletionOptions(trigger_characters=[".", '"', ":"]))
@_safely
def _completion(params: types.CompletionParams) -> list[types.CompletionItem]:
    position = (params.position.line, params.position.character)
    return [
        types.CompletionItem(
            label=c.label,
            kind=_COMPLETION_KINDS.get(c.kind),
            detail=c.detail or None,
            text_edit=types.TextEdit(range=_range(c.replace), new_text=c.label) if c.replace else None,
            documentation=c.documentation or None,
        )
        for c in _document(params.text_document.uri).completions(position)
    ]


def serve() -> None:
    """Serve LSP over stdin and stdout."""
    server.start_io()


__all__ = ["serve", "server"]
