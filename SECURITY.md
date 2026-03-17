# Security Notes

This server can expose both read and write MCP tools over local or containerized transports. Treat that as a real capability boundary.

## Main Rules

- Keep config files mounted read-only whenever possible.
- Keep removable or private store roots mounted read-only unless you explicitly want mutation.
- Use per-store `access_mode` to prevent mutation tools from targeting protected stores.
- If more than one writable store is active, require an explicit `store_id` for mutation tools.
- Enable MCP write-tool auth before exposing write tools beyond a trusted local environment.

## Threat Model

The model provider does not directly edit your RWIF files. The local MCP server does. If a connected MCP client can call write tools and the underlying mount is writable, the local machine can mutate the store.

## Safe Default

The safest default deployment is:

- config mounted read-only
- all stores mounted read-only
- write tools disabled or auth-protected

The writable Docker Desktop example in this repo is intentionally narrow: only the main always-on store root is writable, while config and removable/private roots remain read-only.