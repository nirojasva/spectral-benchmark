import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes



# Genera una clave AES y un vector de inicialización (IV)
def generar_clave():
    clave = get_random_bytes(32)  # AES-256
    iv = get_random_bytes(16)  # Bloque de 16 bytes
    with open("clave.key", "wb") as clave_file:
        clave_file.write(clave)
    with open("iv.key", "wb") as iv_file:
        iv_file.write(iv)

# Cifra un archivo usando AES en modo CBC
def cifrar_archivo(nombre_archivo, clave, iv):
    cipher = AES.new(clave, AES.MODE_CBC, iv)
    
    with open(nombre_archivo, "rb") as file:
        datos = file.read()
    
    datos_cifrados = cipher.encrypt(pad(datos, AES.block_size))
    
    with open(nombre_archivo + ".cifrado", "wb") as file_cifrado:
        file_cifrado.write(datos_cifrados)
    
    os.remove(nombre_archivo)

# Cifra todos los archivos en una carpeta
def cifrar_carpeta(ruta_carpeta):

    with open("clave.key", "rb") as clave_file:
        clave = clave_file.read()
    with open("iv.key", "rb") as iv_file:
        iv = iv_file.read()
    
    # Recorre la carpeta y cifra cada archivo
    for root, dirs, files in os.walk(ruta_carpeta):
        for file in files:
            ruta_completa = os.path.join(root, file)
            cifrar_archivo(ruta_completa, clave, iv)
    print("clave: ", clave)
    print("iv: ", iv)




# Imprimir el directorio de trabajo actual
print("Directorio actual de encriptacion:", os.getcwd())
if open("clave.key", "rb")==False:
    generar_clave()  # Esto solo es necesario la primera vez
cifrar_carpeta(r"/home/nicolas/Downloads/RecordsALPS_V3 - with GT")
