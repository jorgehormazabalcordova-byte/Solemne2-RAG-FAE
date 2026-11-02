import json
import re
import pandas as pd
import psycopg2
import os
from psycopg2.extras import Json
from openai import OpenAI

# 1. Configura tu API Key
client = OpenAI(api_key="TU_API_KEY_AQUI")
def limpiar_texto(texto):
    """Preprocesamiento NLP para limpiar caracteres sucios y espacios."""
    if pd.isna(texto):
        return ""
    texto = str(texto)
    texto = re.sub(r'[\r\n\t]+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def procesar_matrices_rag(ruta_excel):
    xls = pd.ExcelFile(ruta_excel)
    chunks_vectoriales = []
    
    # Asegúrate de que el nombre de la hoja coincida con tu Excel
    nombre_hoja = xls.sheet_names[0] 
    df_trans = pd.read_excel(xls, sheet_name=nombre_hoja)
    
    for idx, row in df_trans.iterrows():
        # Extracción segura de columnas usando .get() y preprocesamiento NLP
        seccion = limpiar_texto(row.get('Sección', ''))
        autor = limpiar_texto(row.get('Referencia', ''))
        aporte = limpiar_texto(row.get('Aporte', ''))
        url = limpiar_texto(row.get('Acceso virtual', ''))
        
        if not seccion and not autor:
            continue
            
        # Lógica de asignación de Categoría Taxonómica
        seccion_lower = seccion.lower()
        if "normativo" in seccion_lower or "ley" in seccion_lower or "sename" in seccion_lower:
            cat_id = "CAT-01"
        elif "autocuidado" in seccion_lower or "duelo" in seccion_lower or "compromiso" in seccion_lower:
            cat_id = "CAT-02"
        elif "preparación" in seccion_lower or "requisitos" in seccion_lower:
            cat_id = "CAT-04"
        elif "niño" in seccion_lower or "experiencia" in seccion_lower:
            cat_id = "CAT-05"
        else:
            cat_id = "CAT-03"
            
        # Chunk Semántico Enriquecido
        page_content = f"""[Sección Taxonómica]: {seccion} (Categoría: {cat_id})
[Referencia Bibliográfica]: {autor}
[Objetivo o Aporte]: {aporte}
[Enlace de Verificación]: {url}"""
        
        metadata = {
            "chunk_id": f"ref_{idx+1}",
            "categoria": cat_id,
            "seccion": seccion,
            "fuente": "Matriz Bibliográfica FAE"
        }
        
        chunks_vectoriales.append({
            "page_content": page_content,
            "metadata": metadata
        })
    return chunks_vectoriales

def generar_embedding(texto):
    response = client.embeddings.create(
        input=[texto],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

if __name__ == "__main__":
    # 2. Configura el nombre exacto de tu archivo Excel
    archivo_fuente = 'Matriz_referencias_virtuales_FAE_transiciones_adopcion (1).xlsx'
    dataset = procesar_matrices_rag(archivo_fuente)
    print(f"Total de chunks procesados y categorizados: {len(dataset)}")
    
    # 3. Configura tus credenciales de PostgreSQL
    conexion = psycopg2.connect(
        host="localhost",
        database="rag_fae", 
        user="postgres",
        password="TU_PASSWORD_AQUI",
        port=5432,
        client_encoding="utf8"  # <--- Agrega esta línea con la coma arriba
    )
    cursor = conexion.cursor()
    print("Generando embeddings e insertando en PostgreSQL con pgvector...")
    
    for item in dataset:
        contenido = item['page_content']
        meta = item['metadata']
        vector_embedding = generar_embedding(contenido)
        
        sql = """
        INSERT INTO vectores_rag_fae 
        (chunk_id, categoria_id, seccion, fuente, page_content, metadata_json, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            meta.get('chunk_id'),
            meta.get('categoria'),
            meta.get('seccion'),
            meta.get('fuente'),
            contenido,
            Json(meta),
            vector_embedding
        ))
        
    conexion.commit()
    cursor.close()
    conexion.close()
    print("¡Ingesta y vectorización completada exitosamente!")
    