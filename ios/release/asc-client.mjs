// App Store Connect API client for Mishkefet's release scripts. Uses the PERSONAL team key
// (Itay Inbar, team 65MGR94YVU) from ~/.appstoreconnect/itay-personal-team, never the repo,
// and never the root ~/.appstoreconnect/key_id (that belongs to other setups).
import { createPrivateKey, sign } from 'node:crypto';
import { readFileSync } from 'node:fs';

const DIR = `${process.env.HOME}/.appstoreconnect/itay-personal-team`;
export const KEY_ID = process.env.ASC_KEY_ID || readFileSync(`${DIR}/key_id`, 'utf8').trim();
export const ISSUER = process.env.ASC_ISSUER_ID || readFileSync(`${DIR}/issuer`, 'utf8').trim();
export const KEY_PATH = `${DIR}/AuthKey_${KEY_ID}.p8`;
const KEY = createPrivateKey(readFileSync(KEY_PATH));

function token() {
  const now = Math.floor(Date.now() / 1000);
  const enc = (v) => Buffer.from(JSON.stringify(v)).toString('base64url');
  const unsigned = `${enc({ alg: 'ES256', kid: KEY_ID, typ: 'JWT' })}.${enc({ iss: ISSUER, iat: now, exp: now + 1100, aud: 'appstoreconnect-v1' })}`;
  return `${unsigned}.${sign('sha256', Buffer.from(unsigned), { key: KEY, dsaEncoding: 'ieee-p1363' }).toString('base64url')}`;
}

export async function api(method, path, body) {
  const response = await fetch(path.startsWith('http') ? path : `https://api.appstoreconnect.apple.com${path}`, {
    method, headers: { Authorization: `Bearer ${token()}`, 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined,
  });
  const text = await response.text();
  const json = text ? JSON.parse(text) : null;
  if (response.status >= 400) {
    const err = json?.errors?.map((e) => `${e.title}: ${e.detail}`).join(' | ') || text.slice(0, 300);
    throw new Error(`${method} ${path} failed: ${response.status} ${err}`);
  }
  return json;
}

export async function appByBundle(bundleId) {
  return (await api('GET', `/v1/apps?filter[bundleId]=${bundleId}`)).data[0] ?? null;
}
