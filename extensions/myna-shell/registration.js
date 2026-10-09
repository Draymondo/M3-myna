// registration.js - the host's "a pill is coming" registration with the
// daemon (`RegisterClient`/`UnregisterClient`).
//
// The daemon suppresses its notification fallback while any client is
// registered, and the renderer needs seconds to start and register itself.
// So the host registers the moment the daemon's name appears, before it
// spawns the renderer, and withdraws when it stops intending to show a pill:
// the host gave up on the renderer, or the extension is disabled. The daemon
// prunes a vanished unique name by itself, so a Shell crash needs nothing, and
// its registry dies with it, so a vanished daemon needs nothing either.
//
// Only Gio, so the contract is testable outside a running Shell.

const METHOD_REGISTER = 'RegisterClient';
const METHOD_UNREGISTER = 'UnregisterClient';

export class PillRegistration {
    /**
     * @param {object} deps
     * @param {DictationProxy} deps.proxy - the shared daemon proxy.
     * @param {function(string):void} [deps.log]
     */
    constructor({proxy, log = () => {}}) {
        this._proxy = proxy;
        this._log = log;
        this._registered = false;
    }

    get registered() {
        return this._registered;
    }

    /** The daemon's name appeared: say a pill is coming. Idempotent. */
    register() {
        if (this._registered)
            return;
        if (this._send(METHOD_REGISTER))
            this._registered = true;
    }

    /** No pill is coming any more: withdraw. A no-op when not registered. */
    unregister() {
        if (!this._registered)
            return;
        this._registered = false;
        this._send(METHOD_UNREGISTER);
    }

    /** The daemon went away, and its registry with it: forget, say nothing. */
    forget() {
        this._registered = false;
    }

    _send(method) {
        const proxy = this._proxy.proxy;
        if (!proxy)
            return false;
        try {
            proxy.call(method, null, 0, -1, null, (source, res) => {
                try {
                    source.call_finish(res);
                } catch (e) {
                    this._log(`${method} failed: ${e.message ?? e}`);
                }
            });
        } catch (e) {
            this._log(`${method} failed: ${e.message ?? e}`);
            return false;
        }
        return true;
    }
}
