# NACHGPT integration contract for dental-llm

Status: isolated design contract. This does not connect dental-llm to NACHGPT production.

## Core rule

NACHGPT owns identity, authorization, tenant scope, and fact collection.
dental-llm only reasons over the verified read-only fact packet supplied by NACHGPT.

The model never:
- resolves user identity from browser text,
- expands role permissions,
- selects another tenant,
- reads production stores directly,
- writes to R2/Postgres/Dropbox,
- executes CAD/CAM or machine actions,
- substitutes generic knowledge when operational evidence is missing.

## Server identity authority

Use the existing NACHGPT server-side identity stack:
- `app/services/nachgpt_identity_service.py`
- `app/services/admin_authorization_service.py`
- `app/services/platform_identity_service.py`

Business boundary already enforced there:
- Platform owner / global authority: platform owner or platform admin only.
- Tenant CEO/owner/admin: own laboratory only.
- Employee/supervisor: role-scoped inside own laboratory.
- Client/doctor: own authorized cases only.

## Fact packet sources

The integration layer in NACHGPT should build categories only from existing canonical read services.

| Fact category | Existing NACHGPT source candidates | Notes |
|---|---|---|
| `case` / `own_case` | `canonical_case_resolver_service.py`, `canonical_store_merge_service.py`, `case_authorization_service.py` | Reauthorize before read. |
| `case_files` / `own_case_files` | `storage_case_objects_service.list_case_objects`, case master read services | R2 physical truth where applicable. |
| `processor` + `intake` | `dental_ai_intake_health_service.build_intake_health` | Read-only durable intake evidence. |
| `r2_evidence` | `storage_case_objects_service.list_case_objects`, physical R2 inventory helpers | R2 is source of truth. |
| `backup` | `dental_ai_r2_dropbox_health_service.build_r2_dropbox_health` | Dropbox remains backup-only. |
| `quality` | `quality_dashboard_service.build_quality_dashboard` with tenant reauthorization | No cross-tenant aggregation for tenant roles. |
| `delivery` | `overdue_alerts_service.build_overdue_alerts` with tenant filtering | Client gets only own-case tracking. |
| `finance` | existing finance / CEO report services | Tenant CEO only own lab; platform owner may cross labs when explicitly authorized. |
| `executive_reports` | `ceo_daily_report_service.build_ceo_daily_report`, `ceo_ai_executive_brief_service.build_ceo_ai_fact_packet` | Report facts remain authoritative. |
| `platform_health` / `global_labs` | `dental_ai_platform_owner_service.build_platform_owner_read_context` | SUPER_ADMIN + explicit `global_labs` only. |
| `inventory` | tenant-scoped inventory services | No automatic purchase/write actions. |
| `hyperdent`, `blender`, `exocad` | canonical case/files/production adapters for those flows | Read-only only; frozen Windows 1.0.17 remains untouched. |

## Request contract

NACHGPT -> dental-llm:

```json
{
  "pregunta": "Audita la ultima corrida del procesador",
  "idioma": "es",
  "principal": {
    "user_id": "0578",
    "display_name": "Ignacio Ramirez",
    "role": "SUPER_ADMIN",
    "laboratory_id": "",
    "capabilities": ["global_labs", "processor", "intake"]
  },
  "fact_packet": {
    "verified": true,
    "read_only": true,
    "laboratory_id": "LAB001",
    "target_laboratory_id": "LAB001",
    "scope": "LAB",
    "categories": {
      "processor": {},
      "intake": {}
    },
    "sources": ["postgres_cases", "processor_run"],
    "generated_at": "..."
  }
}
```

The server-to-server request must include `X-NACHGPT-Gateway-Token`.
The token is transport authentication only; it never grants role or tenant authority.

## Fail-closed behavior

Reject before model execution when:
- identity is incomplete,
- role is unknown,
- capabilities are missing,
- requested fact categories exceed role ceiling,
- fact packet is not verified,
- fact packet is not read-only,
- tenant mismatches,
- client scope is not `OWN_CASE`,
- client ownership was not reverified,
- operational question requires evidence categories absent from the packet.

Operational missing evidence returns a deterministic error such as:
`missing_verified_fact_categories:processor,intake`

The NACHGPT integration layer may translate that into a user-facing:
`NO VERIFICADO: faltan fuentes necesarias para responder con seguridad.`

## Client policy

CLIENT/doctor must never receive:
- other clients' cases,
- employee data,
- laboratory-wide production totals,
- internal R2 keys or infrastructure details,
- Postgres/Dropbox internals,
- internal finance,
- machine queues,
- admin or platform information.

Only own authorized case facts are eligible.

## Initial release policy

All NACHGPT operational use is READ ONLY.
No action broker, no writes, no machine dispatch, no queue mutation, no file deletion, no case mutation.
Permissions can only be expanded after sustained audit accuracy is demonstrated.
