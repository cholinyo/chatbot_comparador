# -----------------------------------------------
# limpiar_git_historial.ps1
# Limpia archivos grandes (.env y models/) del historial Git con BFG
# -----------------------------------------------

Write-Host "🔍 Buscando archivos mayores de 100MB..."
$archivos_grandes = Get-ChildItem -Recurse -File | Where-Object { $_.Length -gt 100MB }
if ($archivos_grandes.Count -eq 0) {
    Write-Host "✅ No se encontraron archivos mayores a 100MB en el sistema de archivos."
} else {
    Write-Host "⚠️ Archivos grandes encontrados:"
    $archivos_grandes | ForEach-Object {
        Write-Host "$($_.FullName) - $([math]::Round($_.Length / 1MB, 2)) MB"
    }
}

# Descargar BFG si no está
$bfg_jar = "bfg-1.14.0.jar"
if (-Not (Test-Path $bfg_jar)) {
    Write-Host "`n⬇️ Descargando BFG Repo-Cleaner..."
    Invoke-WebRequest -Uri "https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar" -OutFile $bfg_jar
} else {
    Write-Host "`n✅ BFG ya descargado."
}

# Aplicar BFG para eliminar archivos del historial
Write-Host "`n🚀 Ejecutando BFG para eliminar models/ y .env del historial..."
java -jar $bfg_jar --delete-folders models --delete-files .env --no-blob-protection

# Limpiar y compactar el repositorio
Write-Host "`n🧹 Limpiando referencias antiguas..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Forzar el push limpio
Write-Host "`n📤 Forzando push limpio al repositorio remoto..."
git push origin --force

Write-Host "`n✅ Proceso completado. Tu repositorio ha sido limpiado y subido."
