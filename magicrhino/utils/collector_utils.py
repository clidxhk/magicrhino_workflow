import sqlite3
import os
import json
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd

# Default field configuration
DEFAULT_FIELDS = {
    'smiles': {'name': 'SMILES', 'type': 'text', 'required': True, 'enabled': True, 'is_default': True},
    'doi': {'name': 'DOI', 'type': 'text', 'required': False, 'enabled': True, 'is_default': True},
    'notes': {'name': 'Notes', 'type': 'textarea', 'required': False, 'enabled': True, 'is_default': True},
}

def save_field_config(config, db_path):
    """Save field configuration to database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS field_config (id INTEGER PRIMARY KEY, config TEXT)')
    cursor.execute('DELETE FROM field_config')
    cursor.execute('INSERT INTO field_config (config) VALUES (?)', (json.dumps(config),))
    conn.commit()
    conn.close()

def load_field_config(db_path):
    """Load field configuration from database"""
    if not os.path.exists(db_path):
        return DEFAULT_FIELDS.copy()
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT config FROM field_config ORDER BY id DESC LIMIT 1')
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
    except:
        pass
    
    return DEFAULT_FIELDS.copy()

def create_database(db_name):
    """Create new database"""
    if not db_name.endswith('.db'):
        db_name += '.db'
    
    if os.path.exists(db_name):
        return False, f"Database {db_name} already exists"
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE molecules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_smiles TEXT UNIQUE NOT NULL,
                original_smiles TEXT NOT NULL,
                first_input_time TEXT NOT NULL,
                duplicate_count INTEGER DEFAULT 1,
                custom_fields TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE duplicate_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_smiles TEXT NOT NULL,
                input_smiles TEXT NOT NULL,
                input_time TEXT NOT NULL,
                custom_fields TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE TABLE field_config (id INTEGER PRIMARY KEY, config TEXT)')
        
        conn.commit()
        conn.close()
        return True, f"Successfully created: {db_name}"
    except Exception as e:
        return False, f"Creation failed: {str(e)}"

def load_molecules_from_db(db_path):
    """Load molecular data from database"""
    if not os.path.exists(db_path):
        return {}
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='molecules'")
    if not cursor.fetchone():
        conn.close()
        return {}
    
    # Load molecular data
    cursor.execute('''
        SELECT canonical_smiles, original_smiles, first_input_time, duplicate_count, 
               COALESCE(custom_fields, '{}') as custom_fields
        FROM molecules
    ''')
    molecules = cursor.fetchall()
    
    molecules_db = {}
    for canonical_smiles, original_smiles, first_input_time, duplicate_count, custom_fields in molecules:
        custom_data = json.loads(custom_fields) if custom_fields else {}
        
        molecules_db[canonical_smiles] = {
            'original_smiles': original_smiles,
            'timestamp': first_input_time,
            'count': duplicate_count,
            'custom_fields': custom_data,
            'duplicate_smiles': []
        }
    
    # Load duplicate records
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='duplicate_inputs'")
    if cursor.fetchone():
        cursor.execute('''
            SELECT canonical_smiles, input_smiles, input_time, 
                   COALESCE(custom_fields, '{}') as custom_fields
            FROM duplicate_inputs ORDER BY input_time
        ''')
        duplicates = cursor.fetchall()
        
        for canonical_smiles, input_smiles, input_time, custom_fields in duplicates:
            if canonical_smiles in molecules_db:
                custom_data = json.loads(custom_fields) if custom_fields else {}
                molecules_db[canonical_smiles]['duplicate_smiles'].append({
                    'smiles': input_smiles,
                    'timestamp': input_time,
                    'custom_fields': custom_data
                })
    
    conn.close()
    return molecules_db

def save_molecule(canonical_smiles, original_smiles, timestamp, custom_fields, db_path):
    """Save new molecule"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO molecules (canonical_smiles, original_smiles, first_input_time, custom_fields)
        VALUES (?, ?, ?, ?)
    ''', (canonical_smiles, original_smiles, timestamp, json.dumps(custom_fields)))
    
    conn.commit()
    conn.close()

def save_duplicate(canonical_smiles, input_smiles, input_time, custom_fields, db_path):
    """Save duplicate record"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO duplicate_inputs (canonical_smiles, input_smiles, input_time, custom_fields)
        VALUES (?, ?, ?, ?)
    ''', (canonical_smiles, input_smiles, input_time, json.dumps(custom_fields)))
    
    cursor.execute('''
        UPDATE molecules SET duplicate_count = duplicate_count + 1
        WHERE canonical_smiles = ?
    ''', (canonical_smiles,))
    
    conn.commit()
    conn.close()

def canonicalize_smiles(smiles):
    """Canonicalize SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        return Chem.MolToSmiles(mol), mol
    except:
        return None, None

def mol_to_svg(mol, size=(300, 300)):
    """Generate an editable SVG molecular structure."""
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        opts = drawer.drawOptions()
        opts.bondLineWidth = 1.6
        opts.clearBackground = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


def mol_to_image(mol, size=(300, 300)):
    """Generate molecular SVG image. Kept for backward-compatible imports."""
    return mol_to_svg(mol, size=size)

def process_field_values(custom_fields):
    """Process field values"""
    processed = {}
    for key, value in custom_fields.items():
        if isinstance(value, bool):
            processed[key] = value
        elif hasattr(value, 'strftime'):
            processed[key] = value.strftime("%Y-%m-%d")
        else:
            processed[key] = value
    return processed

def check_and_add_molecule(smiles, custom_fields, molecules_db, db_path):
    """Check and add molecule"""
    canonical_smiles, mol = canonicalize_smiles(smiles)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if canonical_smiles is None:
        return 'invalid', f"Invalid SMILES: {smiles}", None, None
    
    processed_fields = process_field_values(custom_fields)
    
    if canonical_smiles in molecules_db:
        # Molecular structure already exists
        molecules_db[canonical_smiles]['count'] += 1
        molecules_db[canonical_smiles]['duplicate_smiles'].append({
            'smiles': smiles,
            'timestamp': timestamp,
            'custom_fields': processed_fields
        })
        save_duplicate(canonical_smiles, smiles, timestamp, processed_fields, db_path)
        return 'duplicate', f"Duplicate molecule, saved: {canonical_smiles}", canonical_smiles, mol
    else:
        # New molecule
        molecules_db[canonical_smiles] = {
            'original_smiles': smiles,
            'timestamp': timestamp,
            'count': 1,
            'custom_fields': processed_fields,
            'duplicate_smiles': []
        }
        save_molecule(canonical_smiles, smiles, timestamp, processed_fields, db_path)
        return 'added', f"Successfully added: {canonical_smiles}", canonical_smiles, mol

def export_data(molecules_db, field_config):
    """Export data"""
    data = []
    
    for canonical_smiles, info in molecules_db.items():
        # Main record
        row = {
            'Canonical SMILES': canonical_smiles,
            'Original SMILES': info['original_smiles'],
            'Input Time': info['timestamp'],
            'Duplicate Count': info['count'],
            'Record Type': 'Main Record'
        }
        
        # Add custom fields
        for field_key, field_info in field_config.items():
            if field_info['enabled'] and field_key != 'smiles':
                value = info.get('custom_fields', {}).get(field_key, '')
                if value is None:
                    value = ''
                elif isinstance(value, bool):
                    value = 'Yes' if value else 'No'
                elif not isinstance(value, str):
                    value = str(value)
                row[field_info['name']] = value
        
        data.append(row)
        
        # Duplicate records
        for dup_info in info['duplicate_smiles']:
            dup_row = {
                'Canonical SMILES': canonical_smiles,
                'Original SMILES': dup_info['smiles'],
                'Input Time': dup_info['timestamp'],
                'Duplicate Count': info['count'],
                'Record Type': 'Duplicate Record'
            }
            
            # Add custom fields
            for field_key, field_info in field_config.items():
                if field_info['enabled'] and field_key != 'smiles':
                    value = dup_info.get('custom_fields', {}).get(field_key, '')
                    if value is None:
                        value = ''
                    elif isinstance(value, bool):
                        value = 'Yes' if value else 'No'
                    elif not isinstance(value, str):
                        value = str(value)
                    dup_row[field_info['name']] = value
            
            data.append(dup_row)
    
    return pd.DataFrame(data)