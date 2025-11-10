import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad


path = r'/home/nicolas/spectral_anomaly_detector/datasets/raw/RecordsALPS/RecordsALPS_V3 - with GT' 
# Descifra un archivo usando AES en modo CBC
def descifrar_archivo(nombre_archivo_cifrado, clave, iv):
    cipher = AES.new(clave, AES.MODE_CBC, iv)

    with open(nombre_archivo_cifrado, "rb") as file_cifrado:
        datos_cifrados = file_cifrado.read()

    datos_descifrados = unpad(cipher.decrypt(datos_cifrados), AES.block_size)

    nombre_archivo_descifrado = nombre_archivo_cifrado.replace(".cifrado", "")
    with open(nombre_archivo_descifrado, "wb") as file_descifrado:
        file_descifrado.write(datos_descifrados)

    os.remove(nombre_archivo_cifrado)

# Descifra todos los archivos en una carpeta
def descifrar_carpeta(ruta_carpeta):
    # Lee la clave y el IV
    with open("clave.key", "rb") as clave_file:
        clave = clave_file.read()
    with open("iv.key", "rb") as iv_file:
        iv = iv_file.read()

    # Recorre la carpeta y descifra cada archivo
    for root, dirs, files in os.walk(ruta_carpeta):
        for file in files:
            ruta_completa = os.path.join(root, file)
            if ruta_completa.endswith(".cifrado"):
                descifrar_archivo(ruta_completa, clave, iv)
    print("clave: ", clave)
    print("iv: ", iv)

# Imprimir el directorio de trabajo actual
print("Directorio actual de desencriptacion:", os.getcwd())
# Ejecución de la función
descifrar_carpeta(path)
