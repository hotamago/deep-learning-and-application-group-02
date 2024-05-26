from huggingface_hub import hf_hub_download
model_name = "hotaEfficientNetV2S_super_sigmoid_224x224_v7.keras"
hf_hub_download(repo_id="hotamago/deep-learning-and-application-group-02", filename=model_name, revision="main", repo_type="model", local_dir="model", local_dir_use_symlinks=False)