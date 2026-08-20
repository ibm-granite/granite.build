# Plugins

> **Audience:** developers packaging functionality that ships *outside* the `granite.build` repo — for
> example an organization-specific asset store, secret manager, or compute environment — and want the
> core to discover it at runtime without any code living in the public repo.

A **plugin** is an ordinary pip-installable Python package that contributes implementations to
`granite.build`'s pluggable subsystems. When the plugin is installed alongside `granite.build`, the core
discovers its classes automatically; when it is absent, the core runs exactly as before. There is no code
in the public repo that names the plugin, and no configuration to switch it on beyond installing it.

## How discovery works

Each pluggable subsystem keeps a registry that it populates in two passes:

1. **In-tree scan** — the subsystem imports the implementations that live in its own package directory
   (this is how the built-in environments, URI handlers, etc. register).
2. **Entry-point scan** — the subsystem then enumerates a well-known
   [Python packaging *entry-point group*](https://packaging.python.org/en/latest/specifications/entry-points/)
   and folds any classes it finds into the same registry.

The entry-point table in your plugin's `pyproject.toml` **is** the plugin manifest — there is no separate
manifest file to author. You declare, per subsystem, which of your classes to expose:

```toml
# pyproject.toml of your plugin package (e.g. granite_build_ibm)
[project.entry-points."gbserver.uri_handlers"]
lh = "granite_build_ibm.uri_handlers.lh:LhURI"

[project.entry-points."gbserver.asset_stores"]
lakehouse = "granite_build_ibm.asset_stores.lhstore:Lhstore"

[project.entry-points."gbserver.secret_managers"]
ibmcloud = "granite_build_ibm.secret_managers.ibmcloud:IbmcloudSpaceSecretManager"

[project.entry-points."gbserver.environments"]
myenv = "granite_build_ibm.environments.myenv:Myenv"
```

> **The double-quotes around the group name are required.** These groups contain a dot, and in TOML an
> unquoted dot is a table-nesting separator — `[project.entry-points.gbserver.uri_handlers]` declares two
> nested tables (`gbserver` → `uri_handlers`), *not* an entry-point group named `gbserver.uri_handlers`.
> Quoting the whole dotted name keeps it a single key. (The [entry-points
> spec](https://packaging.python.org/en/latest/specifications/entry-points/) shows dotted group names
> unquoted only because its examples are in the `entry_points.txt` INI format, where section headers like
> `[pygments.styles]` are never quoted; the TOML `pyproject.toml` surface has the opposite requirement.)

The shared discovery helper lives in [`src/gbcommon/plugins.py`](../../src/gbcommon/plugins.py). It wraps
`importlib.metadata.entry_points(group=...)`, loads each entry point, and is **defensive by
construction**: if the entry-point table cannot be read, or a single entry point fails to import, the
failure is logged and skipped so that one broken plugin never prevents the core (or its other plugins)
from starting.

## Precedence: core wins

The in-tree scan runs first; the entry-point scan only ever **adds** to the registry. If a plugin
declares an implementation whose key (URI scheme, environment type name, secret-manager key, or the URI
class an asset store handles) is already registered by the core, the **core implementation is kept** and
the plugin's is skipped with a `WARNING` in the logs. Plugins extend the core; in this release they do not
override it.

Both passes — the in-tree scan and the entry-point scan — file their classes through the same
`PluginRegistrar` (in [`src/gbcommon/plugins.py`](../../src/gbcommon/plugins.py)), so this collision rule
lives in exactly one place. A subsystem constructs a registrar with its own registry dict and a
`keys_of(cls, name)` callback that says which key(s) a class maps to; the registrar owns the rest.

## Supported subsystems

These subsystems discover plugins today. Declare a class in the listed group; the class must subclass the
listed base class and provide the listed key.

| Group | Base class | Key derivation |
|---|---|---|
| `gbserver.uri_handlers` | `gbcommon.uri.uri.URI` | the scheme(s) returned by `get_supported_schemes()` |
| `gbserver.asset_stores` | `gbserver.asset.assetstore.Assetstore` | the URI class(es) returned by `get_supported_uri_classes()` |
| `gbserver.environments` | `gbserver.environment.environment.Environment` | the **entry-point name** (registered under both `name.lower()` and the name exactly as declared) |
| `gbserver.secret_managers` | `SpaceSecretManager` **or** `UserSecretManager` | the **entry-point name** (lowercased). One group feeds both families; each class is routed to the family whose base class it subclasses |

For the URI and asset-store groups the entry-point *name* is cosmetic — the registration key comes from
the class's own method, so name your entry point whatever reads well. For the environment and
secret-manager groups the entry-point *name* is the registration key.

> **Name-keyed groups are case-normalized.** Environments register under both `name.lower()` and the
> entry-point name exactly as you declared it; secret managers under `name.lower()`. Because the declared
> name is preserved verbatim, an entry-point name with internal capitals (e.g. `AWSBatch`) is reachable
> both as `awsbatch` and as `AWSBatch`, so a build referencing `type: AWSBatch` resolves as written.

## Reserved groups (wired in later releases)

The same mechanism will be extended to the following subsystems in subsequent releases. The group names
are reserved now so plugin authors can target stable names; wiring them into the core is tracked
separately.

| Group | Purpose |
|---|---|
| `gbserver.auth_providers` | Additional authentication providers (`AuthProvider` subclasses) |
| `gbserver.resilience_strategies` | Additional retry / resilience strategies (`RetryStrategy` subclasses) |
| `gbserver.builtin_steps` | Additional built-in step directories contributed as step search roots |
| `gbcli.plugins` | Additional `gbcli` subcommands (a module exposing a `cli` click command) |

The canonical list of group-name constants is
[`src/gbcommon/plugins.py`](../../src/gbcommon/plugins.py) — import the constants from there rather than
hardcoding the strings.

## Writing a plugin class

A plugin class is just a normal subclass — the same class you would write in-tree. For example, a URI
handler:

```python
# granite_build_ibm/uri_handlers/lh.py
from gbcommon.uri.uri import URI

class LhURI(URI):
    @staticmethod
    def get_supported_schemes():
        return ["lh"]
    # ... the rest of the URI contract
```

Register it by adding the class to the `gbserver.uri_handlers` group in your `pyproject.toml`:

```toml
# pyproject.toml of granite_build_ibm
[project.entry-points."gbserver.uri_handlers"]
lh = "granite_build_ibm.uri_handlers.lh:LhURI"
```

The entry-point *name* on the left (`lh`) is cosmetic for this group — the registration key comes from
`get_supported_schemes()`, so name it whatever reads well. The value on the right is
`<import.path>:<ClassName>`.

Once your package is installed (`pip install granite_build_ibm`), starting `gbserver` will resolve
`lh://` URIs through `LhURI` with no further configuration.
