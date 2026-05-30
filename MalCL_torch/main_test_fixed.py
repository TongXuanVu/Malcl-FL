# ── TEST MODE: chay truoc khi load data ─────────────────────────────────────
if hasattr(args, 'mode') and args.mode == 'test':
    import glob
    from function import test
    from train import data_task
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    test_ckpt_root = (args.test_checkpoint_dir if args.test_checkpoint_dir else ckpt_dir)
    ckpt_files = sorted(glob.glob(os.path.join(test_ckpt_root, 'ckpt_round*.pth')))
    if not ckpt_files:
        logger.error(f'[TEST] Khong tim thay checkpoint trong: {test_ckpt_root}')
    else:
        logger.info(f'[TEST] Tim thay {len(ckpt_files)} checkpoint(s). Bat dau evaluation...')
        X_train_t, Y_train_t, X_test_t, Y_test_t = dataset(config)
        scaler_t = StandardScaler()
        _test_results = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for _cp in ckpt_files:
            _state = torch.load(_cp, map_location=device, weights_only=False)
            _task   = _state['task']
            _round  = _state['round']
            _global = _state['global_round']
            _C = Classifier(init_classes=config.init_classes)
            # expand if needed
            if _task > 0:
                _C = _C.expand_output_layer(config.init_classes, config.n_inc, _task)
            _C.load_state_dict(_state['classifier_state_dict'])
            _C.to(device)
            _C.eval()
            _, _, _, _X_test_t, _Y_test_t, _test_loader, _ = data_task(
                config, X_train_t, Y_train_t, X_test_t, Y_test_t, scaler_t)
            with torch.no_grad():
                _m = test(config, _C, _test_loader)
            logger.info(
                f'[TEST] {os.path.basename(_cp)} | Task {_task} R {_round} | '
                f"Acc: {_m['accuracy']:.2f}% | F1-Mac: {_m['f1_macro']:.2f}%"
            )
            _test_results.append({'checkpoint': os.path.basename(_cp), 'task': _task,
                                   'round': _round, 'global_round': _global, **_m})
        _df_test = pd.DataFrame(_test_results)
        _test_csv = os.path.join(results_dir, 'test_results.csv')
        _df_test.to_csv(_test_csv, index=False)
        logger.info(f'[TEST] Ket qua luu tai: {_test_csv}')

        # --- Ve bieu do (Giong HFIN/SPCIL) ---
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def save_test_plot(x_vals, y_vals, metric_name, color, marker):
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, f'{color}-{marker}', linewidth=2, markersize=4)
            plt.xlabel('Global Round / Checkpoint Index')
            plt.ylabel(f'{metric_name} (%)' if metric_name != 'Loss' else 'Loss')
            plt.title(f'[TEST - MalCL] {metric_name} over Checkpoints')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            safe_name = metric_name.lower().replace("-", "_")
            plt.savefig(os.path.join(results_dir, f'test_malcl_{safe_name}.png'), dpi=150)
            plt.close()

        def save_combined_plot(x_vals, y_mic, y_mac, y_wei, category_name):
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_mic, 'b-o', label=f'Micro-{category_name}', linewidth=1.5, markersize=3)
            plt.plot(x_vals, y_mac, 'g-s', label=f'Macro-{category_name}', linewidth=1.5, markersize=3)
            plt.plot(x_vals, y_wei, 'r-^', label=f'Weighted-{category_name}', linewidth=1.5, markersize=3)
            plt.xlabel('Global Round / Checkpoint Index')
            plt.ylabel(f'{category_name} (%)')
            plt.title(f'[TEST - MalCL] {category_name} (Micro vs Macro vs Weighted)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            safe_name = category_name.lower().replace("-", "_")
            plt.savefig(os.path.join(results_dir, f'test_malcl_combined_{safe_name}.png'), dpi=150)
            plt.close()

        if _test_results:
            x_axis = [r['global_round'] for r in _test_results]
            save_test_plot(x_axis, [r['accuracy'] for r in _test_results], 'Accuracy', 'b', 'o')
            save_test_plot(x_axis, [r['loss'] for r in _test_results], 'Loss', 'k', 'X')
            save_combined_plot(x_axis, 
                [r.get('precision_micro', 0) for r in _test_results], 
                [r.get('precision_macro', 0) for r in _test_results], 
                [r.get('precision_weighted', 0) for r in _test_results], 'Precision')
            save_combined_plot(x_axis, 
                [r.get('recall_micro', 0) for r in _test_results], 
                [r.get('recall_macro', 0) for r in _test_results], 
                [r.get('recall_weighted', 0) for r in _test_results], 'Recall')
            save_combined_plot(x_axis, 
                [r.get('f1_micro', 0) for r in _test_results], 
                [r.get('f1_macro', 0) for r in _test_results], 
                [r.get('f1_weighted', 0) for r in _test_results], 'F1-Score')
            logger.info(f'[TEST] Da ve bieu do don va ket hop vao: {results_dir}')
    import sys as _sys
    _sys.exit(0)
