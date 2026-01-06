import json
import pandas as pd
import os
import glob

# --- CONFIGURAZIONE ---
DATASETS_ROOT = 'datasets'
OUTPUT_CSV = 'tracking_data.csv'
TARGET_PREFIX = 'out13'

def get_team_info(class_name):
    class_name = str(class_name).lower()
    if "red" in class_name: return "Red", class_name.replace("red_", "")
    elif "white" in class_name: return "White", class_name.replace("white_", "")
    elif "ball" in class_name: return "Ball", "-"
    elif "ref" in class_name: return "Ref", "-"
    else: return "Other", "-"

def process_datasets():
    all_rows = []
    
    # Cerca pattern: datasets/*/train/_annotations.coco.json
    search_path = os.path.join(DATASETS_ROOT, '*', 'train', '_annotations.coco.json')
    json_files = glob.glob(search_path)
    
    if not json_files:
        print(f"❌ Nessun file JSON trovato in {search_path}")
        return

    print(f"🔎 Trovati {len(json_files)} dataset. Elaborazione...")

    for json_path in json_files:
        folder_train = os.path.dirname(json_path)
        folder_action = os.path.dirname(folder_train)
        action_id = os.path.basename(folder_action)
        
        print(f"   📂 Azione: {action_id} ...")
        
        try:
            with open(json_path, 'r') as f: data = json.load(f)
            
            images_map = {img['id']: img['file_name'] for img in data['images']}
            categories_map = {cat['id']: cat['name'] for cat in data['categories']}
            
            count = 0
            for ann in data['annotations']:
                image_id = ann['image_id']
                fname = images_map.get(image_id, "")
                
                if not fname.startswith(TARGET_PREFIX): continue
                
                img_relative_path = os.path.join(folder_train, fname)
                category_id = ann['category_id']
                class_name = categories_map.get(category_id, "Unknown")
                team, number = get_team_info(class_name)
                bbox = ann['bbox']
                
                row = {
                    'action_id': action_id,
                    'frame_filename': fname,
                    'image_path': img_relative_path,
                    'team': team,
                    'number': number,
                    'raw_class': class_name,
                    'x_feet': bbox[0] + (bbox[2] / 2),
                    'y_feet': bbox[1] + bbox[3],
                    'bbox_x': bbox[0],
                    'bbox_y': bbox[1],
                    'bbox_w': bbox[2],
                    'bbox_h': bbox[3]
                }
                all_rows.append(row)
                count += 1
            print(f"      ✅ Fatto: {count} frame.")
            
        except Exception as e:
            print(f"      ❌ Errore: {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.sort_values(by=['action_id', 'frame_filename'])
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 SALVATO: '{OUTPUT_CSV}' con {len(df)} righe.")
    else:
        print("\n⚠️ Nessun dato trovato.")

if __name__ == "__main__":
    process_datasets()