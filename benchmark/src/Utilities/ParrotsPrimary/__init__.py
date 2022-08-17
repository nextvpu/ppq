
def test(model_path = '/home/zhenglongjie/ppq_pplnn/ppq/model/mmlab_model_dynamic_shape/det_model/faster_rcnn.onnx',
        qparam_file = '/home/zhenglongjie/ppq_pplnn/qparam_files/fasterrcnn_quant_act_cuda.json',
        task = 'det',
        metric = 'bbox',
        working_dir = '/home/zhenglongjie/ppq_pplnn/working_dir/faster_rcnn',
        pplnn_binary = '/home/zhenglongjie/ppq_pplnn/pplnn',
        test_ppq = False,
        cfg_path = None,
        dump_data_dir = '/home/zhenglongjie/ppq_pplnn/dump_data/faster_rcnn'
        ):
    assert task in ['det','seg','cls']
    print('test_ppq:', test_ppq)
    if test_ppq:
        assert not cfg_path is None
        quant_config = yaml.load(open(cfg_path, 'r'), Loader=yaml.Loader)
        quant_config = Dict(quant_config)
        quant_config.platform = 'ppl_cuda'
        quant_config.qparams_file = quant_config.qparams_file.replace('trt','cuda')
        quant_config.qparams_file = quant_config.qparams_file.replace('dsp','cuda')
        quant_config.qparams_file = '/home/zhenglongjie/ppq_pplnn/qparam_files/' + quant_config.qparams_file
        ppq_graph = load_from_onnx(model_path)
        quant_config.model = quant_config.model.replace('path_to_ppq','/home/zhenglongjie/ppq_pplnn/ppq')
        quantizer = quantizer_mapping[quant_config.platform](ppq_graph, quant_config, device='cuda:1')
        qparams = json.load(open(quant_config.qparams_file, 'r'))
        quantizer.forward(with_observer=False, qparams=qparams['quant_info'])

    dataset, data_loader, eval_kwargs = eval(f'mm{task}_dataloader')()
    onnx_model = onnx.load(model_path)
    output_names = {output.name : onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[output.type.tensor_type.elem_type] for output in onnx_model.graph.output}
    # specialize for mask_rcnn
    if 'mask' in model_path:
        metric = [metric, 'segm']
    eval_kwargs.update(dict(metric = metric))

    # we need to change to a working dir to prevent output of pplnn from polluting the workspace
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)
    print('model: ', model_path)
    print('qparam: ', qparam_file)

    results = []
    idx = 0
    for _,data in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            if torch.is_tensor(data['img']):
                img = data['img'].numpy()
            elif torch.is_tensor(data['img'][0]):
                img = data['img'][0].numpy()
            else:
                img = data['img'][0].data[0].numpy()
            img_shape = '_'.join(str(i) for i in list(img.shape))
            img.tofile(f'data.bin')
            cmd = f'{pplnn_binary} \
            --onnx-model {model_path} \
            --inputs data.bin \
            --use-cuda \
            --save-outputs \
            --quantization {qparam_file} \
            --in-shapes {img_shape} \
            --quick-select'
            os.system(cmd)
            batch_size = img.shape[0]
            batch_outputs = {name:np.fromfile(f'pplnn_output-{name}.dat',output_names[name]) for name in output_names}
            if test_ppq:
                # compare ppq result with pplnn result by cosine sim
                ppq_input = [torch.from_numpy(img).to('cuda:1')]
                ppq_res = execute(quantizer.graph, *ppq_input, save_memory=True, device='cuda:1')
                compare_results, disagree_indices = compare_ppq_and_cuda_res(ppq_res, batch_outputs, output_names, batch_size)
                print('Batch align result: ','|'.join([f'{output}:{compare_results[output]}' for output in compare_results]))
                
                # dump data where ppq and pplnn disagree
                os.makedirs(f"{dump_data_dir}", exist_ok=True)
                for idx_ in set(disagree_indices):
                    postfix = '_'.join(img_shape.split('_')[1:])
                    img[idx_].tofile(f"{dump_data_dir}/{idx}_{postfix}.bin")
                    idx += 1
                print('total items in disagreement: ', idx)

            if isinstance(data['img_metas'],list):
                img_metas = data['img_metas'][0].data[0]
            else:
                img_metas = data['img_metas'].data[0]
            batch_results = eval(f'mm{task}_process_output')(batch_outputs, batch_size, img_metas, img.shape)
            results.extend(batch_results)
            # if (_ + 1) % 10 == 0:
            #     print('Temp results: ',dataset.evaluate(results, **eval_kwargs))

    return dataset.evaluate(results, **eval_kwargs)