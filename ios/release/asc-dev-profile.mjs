// Development build on Itay's own iPhone without an Apple ID in Xcode: registers the device
// on the personal team if needed, makes sure a development profile for the app includes it
// and the keychain's development certificate, installs the profile, prints its name.
//   node asc-dev-profile.mjs <udid> [bundleId] [certSerial]
import { mkdirSync, writeFileSync } from 'node:fs';
import { api } from './asc-client.mjs';

const [udid, bundleId = 'com.itayinbar.mishkefet', serial] = process.argv.slice(2);
if (!udid) throw new Error('usage: asc-dev-profile.mjs <udid> [bundleId] [certSerial]');
const NAME = 'Mishkefet Development';

let device = (await api('GET', `/v1/devices?filter[udid]=${udid}`)).data[0];
if (!device) {
  device = (await api('POST', '/v1/devices', { data: { type: 'devices', attributes: { name: 'Itay iPhone', udid, platform: 'IOS' } } })).data;
}
const bundle = (await api('GET', `/v1/bundleIds?filter[identifier]=${bundleId}`)).data.find((b) => b.attributes.identifier === bundleId);
const certs = (await api('GET', '/v1/certificates?filter[certificateType]=DEVELOPMENT&limit=50')).data;
const cert = certs.find((c) => !serial || c.attributes.serialNumber === serial);
if (!bundle || !cert) throw new Error('missing bundle id or development certificate');

let profile = (await api('GET', `/v1/profiles?filter[name]=${encodeURIComponent(NAME)}&filter[profileState]=ACTIVE&include=certificates,devices`)).data[0];
const has = (rel, id) => profile?.relationships?.[rel]?.data?.some((x) => x.id === id);
if (profile && !(has('certificates', cert.id) && has('devices', device.id))) {
  await api('DELETE', `/v1/profiles/${profile.id}`);
  profile = null;
}
if (!profile) {
  profile = (await api('POST', '/v1/profiles', { data: { type: 'profiles', attributes: { name: NAME, profileType: 'IOS_APP_DEVELOPMENT' },
    relationships: { bundleId: { data: { type: 'bundleIds', id: bundle.id } }, certificates: { data: [{ type: 'certificates', id: cert.id }] },
      devices: { data: [{ type: 'devices', id: device.id }] } } } })).data;
}
const full = (await api('GET', `/v1/profiles/${profile.id}`)).data;
const dir = `${process.env.HOME}/Library/MobileDevice/Provisioning Profiles`;
mkdirSync(dir, { recursive: true });
writeFileSync(`${dir}/${full.attributes.uuid}.mobileprovision`, Buffer.from(full.attributes.profileContent, 'base64'));
console.log(full.attributes.name);
