/**
 * Whether `s` is a registry UUID, as opposed to an artifact URI or any other id.
 * Shared by every place that tells a registered artifact from a URI-named one.
 */
export function isUUID(s: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(s)
}
