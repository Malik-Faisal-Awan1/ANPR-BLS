// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use tauri_plugin_shell::ShellExt;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Spawn the python sidecar
            let sidecar_command = app.shell().sidecar("backend").unwrap();
            let (mut rx, _child) = sidecar_command
                .spawn()
                .expect("Failed to spawn backend sidecar");
            
            // Log sidecar output (for debugging)
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    println!("Backend: {:?}", event);
                }
            });
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
