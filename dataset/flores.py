from datasets import load_dataset, interleave_datasets

fleurs_langs = ['af_za', 'am_et', 'ar_eg', 'as_in', 'ast_es', 'az_az', 'be_by', 'bg_bg', 'bn_in', 'bs_ba', 'ca_es', 'ceb_ph', 'ckb_iq', 'cmn_hans_cn', 'cs_cz', 'cy_gb', 'da_dk', 'de_de', 'el_gr', 'en_us', 'es_419', 'et_ee', 'fa_ir', 'ff_sn', 'fi_fi', 'fil_ph', 'fr_fr', 'ga_ie', 'gl_es', 'gu_in', 'ha_ng', 'he_il', 'hi_in', 'hr_hr', 'hu_hu', 'hy_am', 'id_id', 'ig_ng', 'is_is', 'it_it', 'ja_jp', 'jv_id', 'ka_ge', 'kam_ke', 'kea_cv', 'kk_kz', 'km_kh', 'kn_in', 'ko_kr', 'ky_kg', 'lb_lu', 'lg_ug', 'ln_cd', 'lo_la', 'lt_lt', 'luo_ke', 'lv_lv', 'mi_nz', 'mk_mk', 'ml_in', 'mn_mn', 'mr_in', 'ms_my', 'mt_mt', 'my_mm', 'nb_no', 'ne_np', 'nl_nl', 'nso_za', 'ny_mw', 'oc_fr', 'om_et', 'or_in', 'pa_in', 'pl_pl', 'ps_af', 'pt_br', 'ro_ro', 'ru_ru', 'sd_in', 'sk_sk', 'sl_si', 'sn_zw', 'so_so', 'sr_rs', 'sv_se', 'sw_ke', 'ta_in', 'te_in', 'tg_tj', 'th_th', 'tr_tr', 'uk_ua', 'umb_ao', 'ur_pk', 'uz_uz', 'vi_vn', 'wo_sn', 'xh_za', 'yo_ng', 'yue_hant_hk', 'zu_za']

langs_to_test = ["en_us", "cmn_hans_cn", "es_419", "fr_fr", "ar_eg", "ru_ru", "de_de", "ja_jp", "pt_br", "hi_in"]

dataset_dict = {}
for lang in fleurs_langs:
    dataset = load_dataset('google/fleurs', lang, split='train', streaming=True)
    dataset_dict[lang] = dataset
    print(f"Loaded {lang} dataset")

# interleave the datasets
interleaved_dataset = interleave_datasets(list(dataset_dict.values()),
                                              stopping_strategy='all_exhausted')

from torch.utils.data import DataLoader

# interleaved_dataset = interleaved_dataset.shuffle(
#     seed=42,
#     buffer_size=10000
# )

def collate_fn(batch):
    return batch

# Create DataLoader
dataloader = DataLoader(
    interleaved_dataset,
    batch_size=8,
    collate_fn=collate_fn,
)
from tqdm import tqdm
import time
# Iterate over the first 1000 examples with tqdm
total_batches = 0
target_batches = 10000

print("\nIterating through dataloader with tqdm...")
start_time = time.time()

# Create tqdm progress bar
pbar = tqdm(total=target_batches)

for i, batch in enumerate(dataloader):
    total_batches += 1
    pbar.update(1)
    
    if total_batches >= target_batches:
        break

pbar.close()

end_time = time.time()
total_time = end_time - start_time
examples_per_second = total_batches / total_time

print(f"\nResults:")
print(f"Iterated through {total_batches} examples in {total_time:.2f} seconds")
print(f"Speed: {examples_per_second:.2f} examples per second")
print(f"Average batch processing time: {total_time / (i+1):.4f} seconds per batch")