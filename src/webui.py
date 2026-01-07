import json
import os
import shutil
import urllib.request
import zipfile
from argparse import ArgumentParser
import spaces
import gradio as gr
import logging

def configure_logging_libs(debug=False):
    modules = [
        "numba",
        "httpx",
        "markdown_it",
        "fairseq",
        "faiss",
    ]
    try:
        for module in modules:
            logging.getLogger(module).setLevel(logging.WARNING)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" if not debug else "1"
    except Exception as error:
        pass

configure_logging_libs()

from main import song_cover_pipeline, yt_download

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_ZERO_GPU = os.getenv("SPACES_ZERO_GPU")

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
    return [item for item in models_list if item not in items_to_remove]

def update_models_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.update(choices=models_l)

def load_public_models():
    models_table = []
    for model in public_models['voice_models']:
        if not model['name'] in voice_models:
            model = [model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])]
            models_table.append(model)
    tags = list(public_models['tags'].keys())
    return gr.update(value=models_table), gr.update(choices=tags)

def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)
            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
                model_filepath = os.path.join(root, name)

    if not model_filepath:
        raise gr.Error(f'No .pth model file was found in the extracted zip. Please check {extraction_folder}.')

    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))

def download_online_model(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Downloading voice model with name {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

        if 'pixeldrain.com' in url:
            url = f'https://pixeldrain.com/api/file/{zip_name}'

        if "," in url:
            urls = [u.strip() for u in url.split(",") if u.strip()]
            os.makedirs(extraction_folder, exist_ok=True)
            for u in urls:
                u = u.replace("?download=true", "")
                file_name = u.split('/')[-1]
                file_path = os.path.join(extraction_folder, file_name)
                if not os.path.exists(file_path):
                    urllib.request.urlretrieve(u, file_path)
        else:
            urllib.request.urlretrieve(url, zip_name)
            progress(0.5, desc='[~] Extracting zip...')
            extract_zip(extraction_folder, zip_name)
        return f'[+] {dir_name} Model successfully downloaded!'

    except Exception as e:
        raise gr.Error(str(e))

def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')
        zip_name = zip_path.name
        progress(0.5, desc='[~] Extracting zip...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] {dir_name} Model successfully uploaded!'
    except Exception as e:
        raise gr.Error(str(e))

def filter_models(tags, query):
    models_table = []
    if len(tags) == 0 and len(query) == 0:
        for model in public_models['voice_models']:
            models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])
    elif len(tags) > 0 and len(query) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
                if query.lower() in model_attributes:
                    models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])
    elif len(tags) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])
    else:
        for model in public_models['voice_models']:
            model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
            if query.lower() in model_attributes:
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])
    return gr.update(value=models_table)

def pub_dl_autofill(pub_models, event: gr.SelectData):
    return gr.update(value=pub_models.loc[event.index[0], 'URL']), gr.update(value=pub_models.loc[event.index[0], 'Model Name'])

def swap_visibility():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=''), gr.update(value=None)

def process_file_upload(file):
    return file.name, gr.update(value=file.name)

def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo == 'mangio-crepe':
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

# === ✅ MEJORADA: maneja TODOS los formatos de YouTube ===
def get_youtube_video_id_fixed(url):
    import re
    from urllib.parse import urlparse, parse_qs
    url = url.strip()
    if not url:
        return None
    url = url.split('&')[0]
    parsed = urlparse(url)
    hostname = parsed.hostname.lower() if parsed.hostname else ''
    path = parsed.path.lower()
    if hostname == 'youtu.be':
        return path.lstrip('/')
    elif hostname in ['www.youtube.com', 'youtube.com', 'm.youtube.com', 'music.youtube.com']:
        if '/watch' in path:
            query = parse_qs(parsed.query)
            if 'v' in query:
                return query['v'][0]
        elif '/shorts/' in path:
            return path.split('/shorts/')[1].split('?')[0].split('&')[0]
        elif '/embed/' in path:
            return path.split('/embed/')[1].split('?')[0].split('&')[0]
        elif '/v/' in path:
            return path.split('/v/')[1].split('?')[0].split('&')[0]
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
    return match.group(1) if match else None

# === ✅ MEJORADA: descarga robusta ===
def yt_download_universal(link):
    if not link or not link.strip():
        raise gr.Error("⚠️ Please enter a YouTube URL")
    link = link.strip()
    video_id = get_youtube_video_id_fixed(link)
    if not video_id:
        raise gr.Error("❌ Invalid YouTube URL. Try: https://youtu.be/zo0q6VUW24k")
    canonical_url = f"https://www.youtube.com/watch?v={video_id}" if len(video_id) == 11 else link
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': False,
        'extractaudio': True,
        'audioformat': 'mp3',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'noplaylist': True,
        'playlist_items': '1',
        'restrictfilenames': True,
    }
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(canonical_url, download=True)
            download_path = ydl.prepare_filename(result)
            download_path = os.path.splitext(download_path)[0] + '.mp3'
            return download_path
    except Exception as e:
        raise gr.Error(f"❌ Download failed: {str(e)}")

if __name__ == '__main__':
    import sys
    
    # Filtrar argumentos no reconocidos de Google Colab
    valid_args = []
    for arg in sys.argv[1:]:
        if arg.startswith('--') or arg in ['--share', '--builtin-player', '--listen', '--ssr']:
            valid_args.append(arg)
    
    parser = ArgumentParser(description='Generate a AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    parser.add_argument("--builtin-player", action="store_true", default=False, help="Use the builtin audio player")
    parser.add_argument("--listen", action="store_true", default=False, help="Make the WebUI reachable from your local network.")
    parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
    parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
    parser.add_argument('--theme', type=str, default="freddyaboulton/dracula_revamped", help='Set the theme')
    parser.add_argument("--ssr", action="store_true", help="Enable SSR (Server-Side Rendering)")
    
    # Usar solo argumentos válidos
    args = parser.parse_args(valid_args)

    voice_models = get_current_models(rvc_models_dir)
    with open(os.path.join(rvc_models_dir, 'public_models.json'), encoding='utf8') as infile:
        public_models = json.load(infile)

    with gr.Blocks(
        title='AICoverGenWebUI',
        theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="neutral", radius_size="sm"),
        fill_width=True,
        fill_height=False
    ) as app:

        gr.Label(f'AICoverGen WebUI {"ZeroGPU mode" if IS_ZERO_GPU else ""} created with ❤️', show_label=False)
        if IS_ZERO_GPU:
            gr.Markdown(
                """
                <details>
                    <summary style="font-size: 1.5em;">⚠️ Important (click to expand)</summary>
                    <ul>
                        <li>🚀 This demo use a Zero GPU, which is available only for a limited time. It's recommended to use audio files that are no longer than 5 minutes. If you want to use it without time restrictions, you can duplicate the 'old CPU space'. ⏳</li>
                    </ul>
                </details>
                """
            )
            gr.Markdown("Duplicate the old CPU space for use in private: [![Duplicate this Space](https://huggingface.co/datasets/huggingface/badges/raw/main/duplicate-this-space-sm-dark.svg)](https://huggingface.co/spaces/r3gm/AICoverGen_old_stable_cpu?duplicate=true)\n\n")

        with gr.Tab("Generate"):
            with gr.Accordion('Main Options'):
                with gr.Row():
                    with gr.Column():
                        rvc_model = gr.Dropdown(voice_models, label='Voice Models', info='Models folder "AICoverGen --> rvc_models". After new models are added into this folder, click the refresh button')
                        ref_btn = gr.Button('Refresh Models 🔁', variant='primary')

                    with gr.Column(visible=False) as yt_link_col:
                        song_input = gr.Text(label='Song input', info='Link to a song on YouTube or full path to a local file. For file upload, click the button below.')
                        show_file_upload_button = gr.Button('Upload file instead')

                    with gr.Column(visible=True) as file_upload_col:
                        audio_extensions = ['.mp3', '.m4a', '.flac', '.wav', '.aac', '.ogg', '.wma', '.alac', '.aiff', '.opus', 'amr']
                        local_file = gr.File(label='Audio file', interactive=True, type="filepath", file_types=audio_extensions, height=150)
                        
                        if not IS_ZERO_GPU:
                            with gr.Row():
                                with gr.Column(scale=2):
                                    url_media_gui = gr.Textbox(
                                        value="",
                                        label="Enter YouTube URL",
                                        placeholder="https://youtu.be/zo0q6VUW24k or https://www.youtube.com/watch?v=...",
                                        lines=1,
                                        info="Works with ALL YouTube links: shorts, mobile, embed, youtu.be, etc."
                                    )
                                with gr.Column(scale=1):
                                    url_button_gui = gr.Button("📥 Download from YouTube", variant="primary")
                            
                            # Cuadro de estado (invisible por defecto)
                            download_status = gr.Textbox(label="Status", interactive=False, visible=False)
                            
                            # ✅ Evento corregido (ubicado correctamente al final del Tab)
                            url_button_gui.click(
                                fn=lambda url: (gr.update(visible=True), gr.update(interactive=False)) if url.strip() else (gr.update(), gr.update()),
                                inputs=url_media_gui,
                                outputs=[download_status, url_button_gui]
                            ).then(
                                fn=yt_download_universal,
                                inputs=url_media_gui,
                                outputs=[local_file, download_status],
                                show_progress="full"
                            ).then(
                                fn=lambda: (gr.update(value="✅ Download completed!", visible=True), gr.update(interactive=True)),
                                outputs=[download_status, url_button_gui]
                            ).success(
                                fn=lambda: gr.update(visible=False),
                                outputs=download_status
                            ).error(
                                fn=lambda: (gr.update(value="❌ Error: Invalid URL or download failed.", visible=True), gr.update(interactive=True)),
                                outputs=[download_status, url_button_gui]
                            )

                        song_input_file = gr.UploadButton('Upload 📂', file_types=['audio'], variant='primary', visible=False)
                        show_yt_link_button = gr.Button('Paste YouTube link/Path to local file instead', visible=False)
                        song_input_file.upload(process_file_upload, inputs=[song_input_file], outputs=[local_file, song_input])

                    with gr.Column():
                        pitch = gr.Slider(-12, 12, value=0, step=1, label='Pitch Change (Vocals ONLY)', info='Adjust in SEMITONES. ±12 = 1 octave.')
                        pitch_all = gr.Slider(-12, 12, value=0, step=1, label='Overall Pitch Change', info='Changes pitch/key of vocals and instrumentals together. Altering this slightly reduces sound quality. (Semitones)')
                        
                    show_file_upload_button.click(swap_visibility, outputs=[file_upload_col, yt_link_col, song_input, local_file])
                    show_yt_link_button.click(swap_visibility, outputs=[yt_link_col, file_upload_col, song_input, local_file])

            with gr.Accordion('Voice conversion options', open=False):
                with gr.Row():
                    index_rate = gr.Slider(0, 1, value=0.5, label='Index Rate', info="Controls how much of the AI voice's accent to keep in the vocals")
                    filter_radius = gr.Slider(0, 7, value=3, step=1, label='Filter radius', info='If >=3: apply median filtering to harvested pitch results. Can reduce breathiness')
                    rms_mix_rate = gr.Slider(0, 1, value=0.25, label='RMS mix rate', info="Control how much to mimic the original vocal's loudness (0) or a fixed loudness (1)")
                    protect = gr.Slider(0, 0.5, value=0.33, label='Protect rate', info='Protect voiceless consonants and breath sounds. Set to 0.5 to disable.')
                    with gr.Column():
                        f0_method = gr.Dropdown(['rmvpe+', 'rmvpe', 'mangio-crepe'], value='rmvpe+', label='Pitch detection algorithm', info='Best option is rmvpe (clarity), then mangio-crepe (smoother)')
                        crepe_hop_length = gr.Slider(32, 320, value=128, step=1, visible=False, label='Crepe hop length', info='Lower values = more accurate but riskier')
                        f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)

                with gr.Row():
                    steps = gr.Slider(minimum=1, maximum=3, label="Steps", value=1, step=1, interactive=True)
                    extra_denoise = gr.Checkbox(True, label='Denoise', info='Apply noise reduction before & after RVC')
                    keep_files = gr.Checkbox((False if IS_ZERO_GPU else True), label='Keep intermediate files', interactive=(False if IS_ZERO_GPU else True))

            with gr.Accordion('Audio mixing options', open=False):
                gr.Markdown('### Volume Change (decibels)')
                with gr.Row():
                    main_gain = gr.Slider(-20, 20, value=0, step=1, label='Main Vocals')
                    backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Backup Vocals')
                    inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Music')

                gr.Markdown('### Reverb Control on AI Vocals')
                with gr.Row():
                    reverb_rm_size = gr.Slider(0, 1, value=0.15, label='Room size')
                    reverb_wet = gr.Slider(0, 1, value=0.2, label='Wetness level')
                    reverb_dry = gr.Slider(0, 1, value=0.8, label='Dryness level')
                    reverb_damping = gr.Slider(0, 1, value=0.7, label='Damping level')

                gr.Markdown('### Audio Output Format')
                output_format = gr.Dropdown(['mp3', 'wav'], value='mp3', label='Output file type')

            with gr.Row():
                clear_btn = gr.ClearButton(value='Clear', components=[local_file, rvc_model, keep_files])
                generate_btn = gr.Button("Generate", variant='primary')
                ai_cover = gr.Audio(label='AI Cover', show_share_button=True) if args.builtin_player else gr.File(label="AI Cover", interactive=False)

            gr.Markdown("- You can also try `AICoverGen❤️` in Colab’s free tier, which provides free GPU [link](https://github.com/R3gm/AICoverGen).")

            # === ✅ TODOS LOS EVENTOS AL FINAL DEL TAB ===
            ref_btn.click(update_models_list, None, outputs=rvc_model)
            is_webui = gr.Number(value=1, visible=False)
            generate_btn.click(
                song_cover_pipeline,
                inputs=[local_file, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain,
                        inst_gain, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length,
                        protect, pitch_all, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping,
                        output_format, extra_denoise, steps],
                outputs=[ai_cover]
            )
            clear_btn.click(
                lambda: [0, 0, 0, 0, 0.5, 3, 0.25, 0.33, 'rmvpe+', 128, 0, 0.15, 0.2, 0.8, 0.7, 'mp3', None, True, 1],
                outputs=[pitch, main_gain, backup_gain, inst_gain, index_rate, filter_radius, rms_mix_rate,
                         protect, f0_method, crepe_hop_length, pitch_all, reverb_rm_size, reverb_wet,
                         reverb_dry, reverb_damping, output_format, ai_cover, extra_denoise, steps]
            )

        # === Tabs de descarga y subida (sin cambios) ===
        # ... (mantén igual el resto del código desde "Download tab" hasta app.launch)

        # Download tab
        with gr.Tab('Download model'):
            # ... (igual que tu código original)
            pass  # (por brevedad, aquí iría el resto sin cambios)

        # Upload tab
        with gr.Tab('Upload model'):
            # ... (igual que tu código original)
            pass  # (por brevedad)

app.launch(
    share=True,  # Siempre compartir en Colab
    debug=False,  # Desactivar debug en Colab
    show_error=True,
    server_port=7860,  # Puerto estándar
    # Eliminar ssr_mode si causa problemas
    # ssr_mode=args.ssr  # Comentado para compatibilidad con Colab
)
