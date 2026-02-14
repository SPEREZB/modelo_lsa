# Búsqueda en Cascada para Lenguaje de Señas

## Descripción

Se ha implementado un sistema de búsqueda en cascada que permite buscar archivos JSON de keypoints de manera inteligente cuando el usuario habla por el micrófono.

## Funcionamiento

Cuando el usuario dice una frase, por ejemplo **"le gusta"**, el sistema realiza la siguiente búsqueda:

### 1. Búsqueda Individual
Primero intenta buscar cada palabra por separado:
- Busca `le.json`
- Busca `gusta.json`

### 2. Búsqueda Combinada (si alguna no se encuentra)
Si alguna de las palabras no se encuentra individualmente, intenta buscar la combinación con guión bajo:
- Busca `le_gusta.json`

### 3. Fallback a Primera Letra
Si ninguna de las opciones anteriores funciona, intenta buscar el archivo JSON de la primera letra:
- Busca `l.json` para "le"
- Busca `g.json` para "gusta"

## Ejemplo de Uso

### Caso 1: Palabras individuales encontradas
Usuario dice: **"casa perro"**
- Busca `casa.json` ✓ (encontrado)
- Busca `perro.json` ✓ (encontrado)
- Resultado: Muestra ambas señas por separado

### Caso 2: Combinación encontrada
Usuario dice: **"le gusta"**
- Busca `le.json` ✗ (no encontrado)
- Busca `gusta.json` ✗ (no encontrado)
- Busca `le_gusta.json` ✓ (encontrado)
- Resultado: Muestra la seña combinada

### Caso 3: Fallback a letras
Usuario dice: **"xyz"**
- Busca `xyz.json` ✗ (no encontrado)
- Busca `x.json` ✓ (encontrado como fallback)
- Resultado: Muestra la letra "x"

## Implementación Técnica

### Backend (Python)
- **Endpoint**: `POST /api/avatar/keypoints_sequence`
- **Archivo**: `app_web/controllers/avatar_controller.py`
- **Función**: `get_keypoints_sequence()`

### Frontend (JavaScript)
- **Archivo**: `app_web/templates/avatar.html`
- **Función**: `processWordsWithCascadeSearch(words)`

## Respuesta del Endpoint

```json
{
  "success": true,
  "sequence": [
    {
      "word": "le_gusta",
      "keypoints": [...],
      "found_as": "combined",
      "original_words": ["le", "gusta"]
    }
  ],
  "not_found": [],
  "available": ["a", "c", "u", "v", "le_gusta"],
  "total_found": 1,
  "total_not_found": 0
}
```

## Tipos de Búsqueda

- **individual**: Palabra encontrada tal cual
- **combined**: Combinación de dos palabras con guión bajo
- **letter**: Fallback a la primera letra de la palabra

## Archivos Modificados

1. `app_web/controllers/avatar_controller.py` - Nuevo endpoint `/keypoints_sequence`
2. `app_web/templates/avatar.html` - Función `processWordsWithCascadeSearch()`
3. `data/keypoints_json/le_gusta.json` - Archivo de ejemplo para pruebas

## Cómo Agregar Nuevas Señas Combinadas

Para agregar una nueva seña combinada, simplemente crea un archivo JSON con el formato:
```
nombre_palabra1_palabra2.json
```

Por ejemplo:
- `me_gusta.json`
- `buenos_dias.json`
- `muchas_gracias.json`

El sistema automáticamente detectará y usará estas combinaciones cuando sea necesario.
