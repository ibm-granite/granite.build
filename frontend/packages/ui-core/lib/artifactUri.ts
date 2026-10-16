// The artifact registry and the lineage index spell a Hugging Face repo
// differently: the registry keeps granite.build's own
// hf://huggingface.co/<type>/<owner>/<repo>[/<rev>[/<path>]], while the index
// keys on the web URL https://huggingface.co/[<type>/]<owner>/<repo>[/tree/<rev>[/<path>]]
// (a model has no type segment there). These helpers translate between the two
// so a URI from either side finds the other. Mirrors gbserver's uri_normalize.

const HF_HOST = 'huggingface.co'
const HF_PREFIX = `hf://${HF_HOST}/`
const WEB_PREFIX = `https://${HF_HOST}/`
const TYPE_SEGMENTS = new Set(['models', 'datasets', 'spaces', 'buckets'])
const REVISION_MARKERS = new Set(['tree', 'blob', 'resolve'])

function hfToWeb(uri: string): string | null {
  const parts = uri.slice(HF_PREFIX.length).split('/').filter(Boolean)
  if (parts.length < 3 || !TYPE_SEGMENTS.has(parts[0])) return null
  const [type, owner, repo, ...rest] = parts
  const segments = type === 'models' ? [owner, repo] : [type, owner, repo]
  if (rest.length) {
    // A bucket has no revision: everything after the repo is a path.
    if (type === 'buckets') segments.push(...rest)
    else segments.push('tree', ...rest)
  }
  return WEB_PREFIX + segments.join('/')
}

function webToHf(uri: string): string | null {
  const parts = uri.slice(WEB_PREFIX.length).split(/[?#]/)[0].split('/').filter(Boolean)
  const type = TYPE_SEGMENTS.has(parts[0]) ? parts.shift()! : 'models'
  if (parts.length < 2) return null
  const [owner, repo, ...rest] = parts
  if (type !== 'buckets' && rest.length && REVISION_MARKERS.has(rest[0])) rest.shift()
  return HF_PREFIX + [type, owner, repo, ...rest].join('/')
}

// The spelling the lineage index keys on.
export function lineageUri(uri: string): string {
  return uri.startsWith(HF_PREFIX) ? hfToWeb(uri) ?? uri : uri
}

// Every spelling the registry may hold for an artifact, the given one first.
export function artifactUriSpellings(uri: string): string[] {
  const other = uri.startsWith(HF_PREFIX) ? hfToWeb(uri) : uri.startsWith(WEB_PREFIX) ? webToHf(uri) : null
  return other && other !== uri ? [uri, other] : [uri]
}
