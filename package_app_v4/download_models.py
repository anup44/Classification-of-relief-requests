import sys
import subprocess
import requests
from tqdm.auto import tqdm
import os
import json
from pathlib import Path
import werkzeug
import tarfile

# print (sys.argv)
download_dir = Path.home().joinpath('.requst_classifier')
download_dir.mkdir(exist_ok=True)

file_dir = Path(__file__).parent.absolute()

with open(Path.joinpath(file_dir, 'config.json'), 'r') as f:
    config_dict = json.load(f)

def download_file_progress_bar(url, out_path='', out_file=None):
    out_path = Path(out_path)
    response = requests.get(url, stream=True)
    assert response.ok
    if out_file:
        file_name = out_file
    else:
        try:
            file_name = werkzeug.parse_options_header(response.headers.get('content-disposition'))[1].get('filename')
        except:
            i = 0
            while ('temp_file{}'.format(i) in os.listdir()):
                i += 1
            file_name = 'temp_file{}'.format(i)
    with tqdm.wrapattr(open(Path.joinpath(out_path, file_name), "wb"), "write", miniters=1,
                    total=int(response.headers.get('content-length', 0)),
                    desc=file_name) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)
    download_file_path = download_dir.joinpath(file_name)
    return download_file_path

def extract_tf_tar_package(package_file, target_folder):
    package_file = Path(package_file)
    if str(package_file).endswith("tar.gz"):
        tar = tarfile.open(str(package_file), "r:gz")
        tar.extractall(target_folder)
        tar.close()

def extract_targz_package(package_file, target_folder):
    package_file = Path(package_file)
    target_folder = Path(target_folder)
    print (f'extracting {package_file} to {target_folder}')
    if package_file.exists():
        if str(package_file).endswith("tar.gz"):
            with tarfile.open(package_file) as tar:
                if target_folder.name == tar.next().name:
                        target_folder = target_folder.parent
            with tqdm.wrapattr(open(package_file, "rb"), "read", miniters=1,
                            total=os.stat(package_file).st_size,
                            desc=str(package_file)) as f_tar:
                # tar = tarfile.open(str(package_file), "r:gz")
                tar = tarfile.open(fileobj=f_tar, mode='r:gz')
                # if target_folder.name == tar.getmembers()[0].name:
                #     target_folder = target_folder.parent
                tar.extractall(target_folder)
                tar.close()
        else:
            raise Exception('File is not a tar.gz')
    else:
        raise FileNotFoundError ('Model file not found at {}'.format(package_file))

def download_models():
    # spacy_model = subprocess.Popen([sys.executable, '-m', 'spacy', 'download', 'en_core_web_lg'])
    # print ('downloading spacy model')
    # spacy_model = spacy_model.communicate()
    for model in config_dict['models']:
        print (model)
        download_path = download_file_progress_bar(url=config_dict['models'][model]['url'], 
                                                    out_path=download_dir, 
                                                    out_file=config_dict['models'][model]['name'])
        config_dict['models'][model]['download_path'] = str(download_path)
        with open(file_dir.joinpath('config.json'), 'w') as f:
            print (config_dict)
            f.write(json.dumps(config_dict, indent=4))

        if 'extracted_dir_name' in config_dict['models'][model]:
            extract_targz_package(config_dict['models'][model]['download_path'], download_dir.joinpath(config_dict['models'][model]['extracted_dir_name']))
            os.remove(config_dict['models'][model]['download_path'])
    # print (spacy_model,'exit')
    # extract_targz_package(config_dict['models']['tf_sentence_encoder']['download_path'], download_dir.joinpath(config_dict['models']['tf_sentence_encoder']['extracted_dir_name']))
    # extract_targz_package(config_dict['models']['en_core_web_lg']['download_path'], download_dir.joinpath(config_dict['models']['en_core_web_lg']['extracted_dir_name']))
if __name__ == '__main__':
    download_models()

