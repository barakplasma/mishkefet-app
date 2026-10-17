// Makes sure an App Store provisioning profile exists for the app, signed by the personal
// team's distribution certificate (the one in the login keychain), and installs it where
// Xcode looks. Prints the profile's name for PROVISIONING_PROFILE_SPECIFIER.
// Lets the release run without an Apple ID signed in to Xcode.
import { mkdirSync, writeFileSync } from 'node:fs';
import { api } from './asc-client.mjs';

const [bundleId = 'com.itayinbar.mishkefet', serial] = process.argv.slice(2);
const NAME = `Mishkefet App Store`;

const bundle = (await api('GET', `/v1/bundleIds?filter[identifier]=${bundleId}`)).data.find((b) => b.attributes.identifier === bundleId);
if (!bundle) throw new Error(`Bundle id ${bundleId} is not registered`);
const certs = (await api('GET', '/v1/certificates?filter[certificateType]=DISTRIBUTION&limit=50')).data;
const cert = certs.find((c) => !serial || c.attributes.serialNumber === serial);
if (!cert) throw new Error(`No distribution certificate${serial ? ` with serial ${serial}` : ''}`);

let profile = (await api('GET', `/v1/profiles?filter[name]=${encodeURIComponent(NAME)}&filter[profileState]=ACTIVE&include=certificates`)).data[0];
if (profile) {
  const certIds = profile.relationships?.certificates?.data?.map((c) => c.id) ?? [];
  if (!certIds.includes(cert.id)) {
    await api('DELETE', `/v1/profiles/${profile.id}`);
    profile = null;
  }
}
if (!profile) {
  profile = (await api('POST', '/v1/profiles', { data: { type: 'profiles', attributes: { name: NAME, profileType: 'IOS_APP_STORE' },
    relationships: { bundleId: { data: { type: 'bundleIds', id: bundle.id } }, certificates: { data: [{ type: 'certificates', id: cert.id }] } } } })).data;
}
const full = (await api('GET', `/v1/profiles/${profile.id}`)).data;
const dir = `${process.env.HOME}/Library/MobileDevice/Provisioning Profiles`;
mkdirSync(dir, { recursive: true });
writeFileSync(`${dir}/${full.attributes.uuid}.mobileprovision`, Buffer.from(full.attributes.profileContent, 'base64'));
console.log(full.attributes.name);
