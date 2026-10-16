"""Game definition CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import PlainTextResponse

from idleframework.export import to_xml, to_yaml
from idleframework.model.game import GameDefinition
from server.game_store import game_store
from server.schemas import ErrorResponse, GameCreateResponse, GameListResponse, GameSummary

router = APIRouter()


@router.get("/", response_model=GameListResponse)
def list_games():
    games = game_store.list_games()
    return GameListResponse(games=[GameSummary(**g) for g in games])


@router.get("/{game_id}")
def get_game(game_id: str):
    game = game_store.get_game(game_id)
    if game is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="game_not_found",
                detail=f"No game with ID '{game_id}' exists",
                status=404,
            ).model_dump(),
        )
    return game.model_dump(mode="json")


@router.post("/", response_model=GameCreateResponse, status_code=201)
def create_game(game: GameDefinition):
    try:
        game_id = game_store.save_game(game)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=ErrorResponse(
                error="name_conflict",
                detail=(f"Name '{game.name}' conflicts with a bundled game"),
                status=409,
            ).model_dump(),
        ) from exc
    return GameCreateResponse(id=game_id, name=game.name)


@router.delete("/{game_id}", status_code=204)
def delete_game(game_id: str):
    if game_store.is_bundled(game_id):
        raise HTTPException(
            status_code=403,
            detail=ErrorResponse(
                error="forbidden",
                detail=f"Cannot delete bundled game '{game_id}'",
                status=403,
            ).model_dump(),
        )
    if not game_store.delete_game(game_id):
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="game_not_found",
                detail=f"No user game with ID '{game_id}' exists",
                status=404,
            ).model_dump(),
        )
    return Response(status_code=204)


@router.get("/{game_id}/schema")
def get_schema(game_id: str):
    game = game_store.get_game(game_id)
    if game is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="game_not_found",
                detail=f"No game with ID '{game_id}' exists",
                status=404,
            ).model_dump(),
        )
    return GameDefinition.model_json_schema()


@router.get("/{game_id}/export")
def export_game(game_id: str, format: str = "yaml"):
    game = game_store.get_game(game_id)
    if game is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="game_not_found",
                detail=f"No game with ID '{game_id}' exists",
                status=404,
            ).model_dump(),
        )
    if format == "yaml":
        return PlainTextResponse(to_yaml(game), media_type="text/yaml")
    elif format == "xml":
        return PlainTextResponse(to_xml(game), media_type="application/xml")
    else:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="invalid_format",
                detail=f"Unsupported format '{format}'. Use 'yaml' or 'xml'.",
                status=400,
            ).model_dump(),
        )


# ---------------------------------------------------------------------------
# Validation without saving (used by the Chinese input editor, /input)
# ---------------------------------------------------------------------------

# Node types that the schema accepts but the simulation engine currently ignores.
_ENGINE_IGNORED_TYPES = {
    "nested_generator", "sacrifice", "achievement", "manager", "converter",
    "end_condition", "unlock_gate", "choice_group", "gate", "queue",
}


@router.post("/validate")
def validate_game(payload: dict):
    """Validate a game definition without saving it.

    Returns {"ok": bool, "errors": [{"loc": str, "msg": str}], "warnings": [...]}.
    """
    from pydantic import ValidationError

    errors: list[dict] = []
    warnings: list[dict] = []
    try:
        GameDefinition.model_validate(payload)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            errors.append({"loc": loc, "msg": err.get("msg", "")})
    except ValueError as exc:  # pragma: no cover - defensive
        errors.append({"loc": "", "msg": str(exc)})

    for i, node in enumerate(payload.get("nodes", []) or []):
        t = node.get("type") if isinstance(node, dict) else None
        if t in _ENGINE_IGNORED_TYPES:
            warnings.append({
                "loc": f"nodes.{i}",
                "msg": f"节点 {node.get('id')!r} 类型 {t} 目前模拟引擎不会生效（只保存，不参与计算）",
            })
    if not errors:
        from idleframework.analysis.static_checks import run_static_checks

        game = GameDefinition.model_validate(payload)
        for w in run_static_checks(game):
            ref = w.get("node_id") or w.get("resource_id") or "/".join(w.get("cycle", []))
            loc = f"node:{ref}" if "node_id" in w else f"check:{w['check']}:{ref}"
            item = {"loc": loc, "msg": w["message"], "kind": w["kind"], "severity": w["severity"]}
            if w["kind"] == "unknown_currency":
                errors.append(item)  # a price/requirement names a non-resource
            else:
                warnings.append(item)
    return {"ok": not errors, "errors": errors, "warnings": warnings}
