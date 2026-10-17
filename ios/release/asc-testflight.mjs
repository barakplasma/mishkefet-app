// Waits for Apple to process an uploaded Mishkefet build, answers export compliance, and makes it
// available to an internal TestFlight group that contains only the account holder (Itay).
import { api, appByBundle } from './asc-client.mjs';

const [buildNumber, bundleId = 'com.itayinbar.mishkefet'] = process.argv.slice(2);
const TESTER = 'itayinbar.me@gmail.com';
const app = await appByBundle(bundleId);
if (!app) throw new Error(`No App Store Connect app for ${bundleId}. Create it first (see ios/release/README.md)`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let build;
for (let i = 0; i < 80; i += 1) {
  build = (await api('GET', `/v1/builds?filter[app]=${app.id}&filter[version]=${buildNumber}&fields[builds]=version,processingState,usesNonExemptEncryption`)).data[0];
  const state = build?.attributes.processingState;
  if (state === 'VALID') break;
  if (state === 'INVALID' || state === 'FAILED') throw new Error(`Build ${buildNumber} failed processing`);
  await sleep(30_000);
}
if (!build) throw new Error(`Build ${buildNumber} never appeared in App Store Connect. Apple likely rejected it during processing; check the "Action needed" email for ITMS errors`);
if (build.attributes.processingState !== 'VALID') throw new Error(`Build ${buildNumber} is still processing; rerun this script later`);

const groups = (await api('GET', `/v1/apps/${app.id}/betaGroups`)).data;
let group = groups.find((g) => g.attributes.name === 'Mishkefet');
if (!group) {
  group = (await api('POST', '/v1/betaGroups', { data: { type: 'betaGroups', attributes: { name: 'Mishkefet', isInternalGroup: true, hasAccessToAllBuilds: true },
    relationships: { app: { data: { type: 'apps', id: app.id } } } } })).data;
}
const members = (await api('GET', `/v1/betaGroups/${group.id}/betaTesters?limit=200&fields[betaTesters]=email`)).data
  .map((t) => t.attributes.email?.toLowerCase());
if (!members.includes(TESTER)) {
  await api('POST', '/v1/betaTesters', { data: { type: 'betaTesters', attributes: { email: TESTER, firstName: 'Itay', lastName: 'Inbar' },
    relationships: { betaGroups: { data: [{ type: 'betaGroups', id: group.id }] } } } });
}
console.log(`Build ${buildNumber} processed and available in TestFlight (group "Mishkefet").`);
