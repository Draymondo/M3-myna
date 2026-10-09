// registration.test.js - the host's "a pill is coming" registration.
//
// Against a real private bus: a fake daemon owns the name and records which
// sender called what, so what is asserted is what the daemon's registry would
// see, not what the host believes it sent.
//
//     gjs -m test/registration.test.js     (from extensions/myna-shell/)

import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import System from 'system';

import {DictationProxy} from '../dictationProxy.js';
import {PillRegistration} from '../registration.js';

let failures = 0;
function check(name, cond) {
    if (cond)
        print(`ok   ${name}`);
    else {
        failures++;
        print(`FAIL ${name}`);
    }
}

const NAME = 'com.canonical.Myna.Dictation';
const PATH = '/com/canonical/Myna/Dictation';
const XML = `<node><interface name='${NAME}'>
  <method name='RegisterClient'><arg type='u' direction='out'/></method>
  <method name='UnregisterClient'><arg type='u' direction='out'/></method>
  <property name='State' type='s' access='read'/>
</interface></node>`;

if (!GLib.getenv('DBUS_SESSION_BUS_ADDRESS')) {
    print('skip no session bus (run under dbus-run-session)');
    System.exit(0);
}

const calls = [];
// A private connection stands in for the daemon; the host talks through the
// shared session connection, as the Shell does.
const bus = Gio.DBusConnection.new_for_address_sync(
    Gio.dbus_address_get_for_bus_sync(Gio.BusType.SESSION, null),
    Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT |
    Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION, null, null);
const info = Gio.DBusNodeInfo.new_for_xml(XML).lookup_interface(NAME);
bus.register_object(PATH, info, (conn, sender, path, iface, method, params, invocation) => {
    calls.push(`${method} ${sender}`);
    invocation.return_value(new GLib.Variant('(u)', [1]));
}, () => new GLib.Variant('s', 'idle'), null);

const loop = new GLib.MainLoop(null, false);
function until(what, done) {
    return new Promise(resolve => {
        const deadline = GLib.get_monotonic_time() + 5e6;
        GLib.timeout_add(GLib.PRIORITY_DEFAULT, 10, () => {
            if (done()) {
                resolve();
                return GLib.SOURCE_REMOVE;
            }
            if (GLib.get_monotonic_time() > deadline) {
                check(`timed out waiting for ${what}`, false);
                resolve();
                return GLib.SOURCE_REMOVE;
            }
            return GLib.SOURCE_CONTINUE;
        });
    });
}

(async () => {
  try {
    bus.call_sync('org.freedesktop.DBus', '/org/freedesktop/DBus',
        'org.freedesktop.DBus', 'RequestName',
        new GLib.Variant('(su)', [NAME, 0]), null, Gio.DBusCallFlags.NONE, -1, null);

    const proxy = new DictationProxy();
    proxy.start();
    await until('the proxy', () => proxy.proxy !== null);
    const registration = new PillRegistration({proxy, log: m => print(`# ${m}`)});

    check('nothing is registered at first', !registration.registered);
    registration.unregister();
    check('withdrawing what was never registered says nothing', calls.length === 0);

    registration.register();
    registration.register();
    check('register is idempotent on the host side', registration.registered);
    await until('RegisterClient', () => calls.length >= 1);
    check('the daemon saw one RegisterClient', calls.length === 1 &&
        calls[0].startsWith('RegisterClient :1.'));

    registration.unregister();
    check('unregister clears the state', !registration.registered);
    await until('UnregisterClient', () => calls.length >= 2);
    check('the daemon saw UnregisterClient from the same sender',
        calls[1] === calls[0].replace('RegisterClient', 'UnregisterClient'));

    registration.register();
    await until('second RegisterClient', () => calls.length >= 3);
    registration.forget();
    check('forget (the daemon went away) clears the state', !registration.registered);
    registration.unregister();
    await new Promise(resolve => GLib.timeout_add(GLib.PRIORITY_DEFAULT, 150, () => {
        resolve();
        return GLib.SOURCE_REMOVE;
    }));
    check('forgetting says nothing to the daemon', calls.length === 3);

    loop.quit();
    System.exit(failures === 0 ? 0 : 1);
  } catch (e) {
    print(`FAIL ${e}\n${e.stack}`);
    System.exit(1);
  }
})();
loop.run();
