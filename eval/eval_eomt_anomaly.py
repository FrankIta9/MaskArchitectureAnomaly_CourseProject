def load_model_weights(lm: LightningModule, weights_path: str, device: torch.device):
    ext = os.path.splitext(weights_path)[1].lower()

    if ext == ".ckpt":
        # Lightning ckpt: fai come prima
        # (lm è già stato costruito con ckpt_path se vuoi, ma qui lo facciamo esplicito)
        ckpt = torch.load(weights_path, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = lm.load_state_dict(state, strict=False)
        print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")
        return

    # .bin / .pth: di solito state_dict puro
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Alcuni salvano solo la network (senza prefisso), altri con "network."
    # Proviamo entrambe.
    missing, unexpected = lm.load_state_dict(state, strict=False)
    if len(missing) > 0 and all(k.startswith("network.") for k in missing):
        # prova a caricare direttamente nella network se il dict è “pulito”
        missing2, unexpected2 = lm.network.load_state_dict(state, strict=False)
        print(f"[BIN->network] missing={len(missing2)} unexpected={len(unexpected2)}")
    else:
        print(f"[BIN] missing={len(missing)} unexpected={len(unexpected)}")
