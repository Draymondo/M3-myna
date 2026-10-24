//! A stand-in kglobalaccel: the slice of `org.kde.KGlobalAccel` Settings and
//! Plasma's Shortcuts settings use, with the daemon's quirks that matter to a
//! client (kglobalacceld 6.6):
//!
//! - registering a desktop entry's component gives it a grabbed `_launch`;
//!   registering `_launch` itself in the same call adds a second, ungrabbed
//!   one in its place, so the key never fires;
//! - a key another action holds is dropped from a new key list;
//! - a key sequence is four ints; KWin aborts on fewer.
//!
//! What the stand-in holds is shared with the test through [`State`].

use std::sync::{Arc, Mutex};

use gio::glib::Variant;
use gio::prelude::*;

const XML: &str = "<node>\
  <interface name='org.kde.KGlobalAccel'>\
    <method name='allMainComponents'><arg type='aas' direction='out'/></method>\
    <method name='doRegister'><arg type='as' direction='in'/></method>\
    <method name='getComponent'>\
      <arg type='s' direction='in'/><arg type='o' direction='out'/>\
    </method>\
    <method name='unregister'>\
      <arg type='s' direction='in'/><arg type='s' direction='in'/>\
      <arg type='b' direction='out'/>\
    </method>\
    <method name='shortcutKeys'>\
      <arg type='as' direction='in'/><arg type='a(ai)' direction='out'/>\
    </method>\
    <method name='setForeignShortcutKeys'>\
      <arg type='as' direction='in'/><arg type='a(ai)' direction='in'/>\
    </method>\
    <method name='globalShortcutsByKey'>\
      <arg type='(ai)' direction='in'/><arg type='(i)' direction='in'/>\
      <arg type='a(ssssssaiai)' direction='out'/>\
    </method>\
    <signal name='yourShortcutsChanged'>\
      <arg type='as'/><arg type='a(ai)'/>\
    </signal>\
  </interface>\
</node>";
const INTERFACE: &str = "org.kde.KGlobalAccel";
const ABORT: &str = "org.example.Abort";

#[derive(Clone, Debug, Default)]
pub struct Action {
    pub name: String,
    pub friendly: String,
    pub keys: Vec<i32>,
    /// Grabbed: its keys fire.
    pub present: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Component {
    pub name: String,
    pub friendly: String,
    pub actions: Vec<Action>,
}

#[derive(Debug, Default)]
pub struct State {
    pub components: Vec<Component>,
    /// How many registrations of a new desktop entry's component go unseen,
    /// as while KSycoca has not looked at the disk since the entry was written.
    pub unseen: u32,
}

impl State {
    fn component(&mut self, name: &str) -> Option<&mut Component> {
        self.components.iter_mut().find(|c| c.name == name)
    }

    pub fn action(&self, component: &str, action: &str) -> Option<&Action> {
        let component = self.components.iter().find(|c| c.name == component)?;
        component.actions.iter().find(|a| a.name == action)
    }

    fn action_mut(&mut self, component: &str, action: &str) -> Option<&mut Action> {
        let component = self.component(component)?;
        component.actions.iter_mut().find(|a| a.name == action)
    }

    fn holder(&self, key: i32) -> Option<&Action> {
        let mut actions = self.components.iter().flat_map(|c| &c.actions);
        actions.find(|action| action.keys.contains(&key))
    }

    fn register(&mut self, id: &[String]) {
        let [component, action, component_friendly, action_friendly] = id else {
            return;
        };
        if self.action(component, action).is_some() {
            return;
        }
        if self.component(component).is_none() {
            let entry = component.ends_with(".desktop");
            if entry && self.unseen > 0 {
                self.unseen -= 1;
                return;
            }
            let launch = entry.then(|| Action {
                name: "_launch".into(),
                friendly: component_friendly.clone(),
                keys: Vec::new(),
                present: true,
            });
            self.components.push(Component {
                name: component.clone(),
                friendly: component_friendly.clone(),
                actions: launch.into_iter().collect(),
            });
        }
        let actions = &mut self.component(component).unwrap().actions;
        // Registering `_launch` itself shadows the grabbed one.
        actions.retain(|a| &a.name != action);
        actions.push(Action {
            name: action.clone(),
            friendly: action_friendly.clone(),
            ..Action::default()
        });
    }
}

/// Each key as the four-int sequence kglobalaccel stores.
pub fn sequences(keys: &[i32]) -> Variant {
    let sequences: Vec<(Vec<i32>,)> = keys.iter().map(|key| (vec![*key, 0, 0, 0],)).collect();
    sequences.to_variant()
}

/// The key of a sequence, or why KWin would abort.
fn key_of(sequence: &Variant) -> Result<i32, String> {
    match sequence
        .child_value(0)
        .get::<Vec<i32>>()
        .unwrap()
        .as_slice()
    {
        [key, 0, 0, 0] => Ok(*key),
        ints => Err(format!("KWin aborts on the key sequence {ints:?}")),
    }
}

/// Serve kglobalaccel on the bus at `address` until it goes.
pub fn serve(address: &str, ready: std::sync::mpsc::Sender<()>, state: Arc<Mutex<State>>) {
    super::serve_as(address, "org.kde.kglobalaccel", ready, |connection| {
        let interface = gio::DBusNodeInfo::for_xml(XML)
            .unwrap()
            .lookup_interface(INTERFACE)
            .unwrap();
        let emitter = connection.clone();
        connection
            .register_object("/kglobalaccel", &interface)
            .method_call(move |_, _, _, _, method, parameters, invocation| {
                let mut state = state.lock().unwrap();
                let id = || parameters.child_value(0).get::<Vec<String>>().unwrap();
                match method {
                    "allMainComponents" => {
                        let names: Vec<Vec<&str>> = state
                            .components
                            .iter()
                            .map(|c| vec![c.name.as_str(), c.friendly.as_str(), "", ""])
                            .collect();
                        invocation.return_value(Some(&(names,).to_variant()));
                    }
                    "doRegister" => {
                        state.register(&id());
                        invocation.return_value(None);
                    }
                    "getComponent" => {
                        let (name,) = parameters.get::<(String,)>().unwrap();
                        if state.component(&name).is_none() {
                            return invocation.return_dbus_error(
                                "org.kde.kglobalaccel.NoSuchComponent",
                                &format!("The component '{name}' doesn't exist."),
                            );
                        }
                        let path = format!("/component/{}", name.replace(['.', '-', ' '], "_"));
                        let path = gio::glib::variant::ObjectPath::try_from(path).unwrap();
                        invocation.return_value(Some(&(path,).to_variant()));
                    }
                    "unregister" => {
                        let (component, action) = parameters.get::<(String, String)>().unwrap();
                        let found = state.component(&component).is_some_and(|c| {
                            let before = c.actions.len();
                            c.actions.retain(|a| a.name != action);
                            c.actions.len() != before
                        });
                        invocation.return_value(Some(&(found,).to_variant()));
                    }
                    "shortcutKeys" => {
                        let id = id();
                        let keys = state.action(&id[0], &id[1]).map(|a| a.keys.clone());
                        let keys = sequences(&keys.unwrap_or_default());
                        invocation.return_value(Some(&Variant::tuple_from_iter([keys])));
                    }
                    "setForeignShortcutKeys" => {
                        let id = id();
                        let wanted: Result<Vec<i32>, String> = parameters
                            .child_value(1)
                            .iter()
                            .map(|s| key_of(&s))
                            .collect();
                        let wanted = match wanted {
                            Ok(keys) => keys,
                            Err(why) => return invocation.return_dbus_error(ABORT, &why),
                        };
                        let Some(action) = state.action_mut(&id[0], &id[1]) else {
                            return invocation.return_value(None);
                        };
                        action.keys.clear();
                        let free: Vec<i32> = wanted
                            .into_iter()
                            .filter(|key| *key != 0 && state.holder(*key).is_none())
                            .collect();
                        state.action_mut(&id[0], &id[1]).unwrap().keys = free.clone();
                        drop(state);
                        let changed = Variant::tuple_from_iter([id.to_variant(), sequences(&free)]);
                        emitter
                            .emit_signal(
                                None,
                                "/kglobalaccel",
                                INTERFACE,
                                "yourShortcutsChanged",
                                Some(&changed),
                            )
                            .unwrap();
                        invocation.return_value(None);
                    }
                    "globalShortcutsByKey" => {
                        let key = match key_of(&parameters.child_value(0)) {
                            Ok(key) => key,
                            Err(why) => return invocation.return_dbus_error(ABORT, &why),
                        };
                        // Like kglobalaccel, the first component with a match.
                        let found = state
                            .components
                            .iter()
                            .map(|c| {
                                let held = c.actions.iter().filter(|a| a.keys.contains(&key));
                                held.map(|a| {
                                    (
                                        a.name.clone(),
                                        a.friendly.clone(),
                                        c.name.clone(),
                                        c.friendly.clone(),
                                        "default",
                                        "Default Context",
                                        a.keys.clone(),
                                        Vec::<i32>::new(),
                                    )
                                })
                                .collect::<Vec<_>>()
                            })
                            .find(|held| !held.is_empty())
                            .unwrap_or_default();
                        invocation.return_value(Some(&(found,).to_variant()));
                    }
                    other => unreachable!("{other}"),
                }
            })
            .build()
            .expect("register the stand-in kglobalaccel");
    });
}

/// The client end of a stand-in kglobalaccel on a fresh bus, and what it
/// holds.
pub fn connect() -> (
    gio::DBusConnection,
    super::Bus,
    std::thread::JoinHandle<()>,
    Arc<Mutex<State>>,
) {
    let state = Arc::new(Mutex::new(State::default()));
    let (connection, bus, thread) = super::connect_with({
        let state = Arc::clone(&state);
        move |address, ready| serve(address, ready, state)
    });
    (connection, bus, thread, state)
}
