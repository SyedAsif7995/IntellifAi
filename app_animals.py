def bind_examples(examples, inputs, outputs, fn):
    gr.Examples(
        examples=examples,
        inputs=inputs,
        outputs=outputs,
        fn=fn,
        cache_examples=False,
        examples_per_page=len(examples)
    )

with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")])) as demo:
    gr.HTML(load_description(title_md))
    gr.Markdown(load_description("assets/gradio/gradio_description_upload_animal.md"))

    with gr.Row():
        with gr.Column():
            with gr.Accordion(open=True, label="🐱 Source Animal Image"):
                source_image_input = gr.Image(type="filepath")
                bind_examples(
                    [[osp.join(example_portrait_dir, f"s{n}.jpg")] for n in [25,30,31,32,33,39,40,41,38,36]],
                    [source_image_input],
                    [],
                    None
                )
            # cropping inputs...

        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("📁 Driving Pickle"):
                    driving_video_pickle_input = gr.File()
                    bind_examples(
                        [[osp.join(example_video_dir, f"{name}.pkl")] for name in
                         ["wink", "shy", "aggrieved", "open_lip", "laugh", "talking", "shake_face"]],
                        [driving_video_pickle_input],
                        [],
                        None
                    )
                with gr.TabItem("🎞️ Driving Video"):
                    driving_video_input = gr.Video()
                    bind_examples(
                        [[osp.join(example_video_dir, f"d{n}.mp4")] for n in [19,14,6,3]],
                        [driving_video_input],
                        [],
                        None
                    )
                tab_selection = gr.Textbox(visible=False)
                tab_pickle.select(lambda: "Pickle", None, tab_selection)
                tab_video.select(lambda: "Video", None, tab_selection)

    # rest of UI unchanged...

    process_button_animation.click(
        fn=gpu_wrapped_execute_video,
        inputs=[source_image_input, driving_video_input, driving_video_pickle_input, flag_do_crop_input, flag_remap_input, driving_multiplier,
                flag_stitching, flag_crop_driving_video_input, scale, vx_ratio, vy_ratio,
                scale_crop_driving_video, vx_ratio_crop_driving_video, vy_ratio_crop_driving_video, tab_selection],
        outputs=[output_video_i2v, output_video_concat_i2v, output_video_i2v_gif],
        show_progress=True,
    )

demo.launch(server_port=args.server_port, share=args.share, server_name=args.server_name)
