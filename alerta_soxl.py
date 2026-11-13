"""Alerta diaria de SOXL con Yahoo Finance y Telegram. No envía órdenes.

Instalación:
    python -m pip install yfinance pandas numpy requests pandas-market-calendars
Uso:
    python alerta_soxl.py            # Solo consola. No necesita credenciales.
    python alerta_soxl.py --enviar   # Sesión de hoy cerrada; requiere variables
                                    # TELEGRAM_TOKEN y TELEGRAM_CHAT_ID.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import timedelta
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf


TICKER = "SOXL"  # NYSE Arca, USD. No usar este calendario para otros mercados.
INICIO_HISTORICO = "2020-01-01"  # Fijo: no se reinicia la estrategia cada día.
ZONA_MERCADO = "America/New_York"
ESPERA_PUBLICACION = pd.Timedelta(minutes=25)
ATR_MULT = 2.5
RSI_ENTRADA = 45.0
ESTADO_LOCAL = Path.home() / ".alerta_soxl" / "avisos.sqlite3"


def calendario_de_sesiones(ahora: pd.Timestamp) -> pd.DataFrame:
    """Incluye sesiones futuras para identificar la próxima apertura."""
    hoy_ny = ahora.tz_convert(ZONA_MERCADO).date()
    return mcal.get_calendar("NYSE").schedule(
        start_date=INICIO_HISTORICO,
        end_date=hoy_ny + timedelta(days=14),
    )


def descargar_historico(sesiones_cerradas: pd.DatetimeIndex) -> pd.DataFrame:
    ultima = sesiones_cerradas[-1]
    df = yf.download(
        TICKER,
        start=INICIO_HISTORICO,
        end=(ultima + pd.Timedelta(days=1)).date().isoformat(),  # Exclusivo.
        interval="1d",
        auto_adjust=False,
        prepost=False,
        ignore_tz=True,
        multi_level_index=False,
        threads=False,
        progress=False,
        timeout=15,
    )
    if df is None or df.empty:
        raise ValueError("Yahoo Finance no devolvió velas. No se generó señal.")

    columnas = ["High", "Low", "Close", "Adj Close"]
    if not set(columnas).issubset(df.columns):
        raise ValueError("Yahoo Finance no devolvió todas las columnas necesarias.")

    df = df[columnas].copy()
    df.index = pd.DatetimeIndex(df.index).normalize()
    if df.index.has_duplicates:
        raise ValueError("El histórico contiene fechas duplicadas.")

    # Reindexar detecta tanto un último cierre atrasado como huecos intermedios.
    df = df.reindex(sesiones_cerradas)
    valores = df.to_numpy(dtype=float)
    invalidas = (~np.isfinite(valores) | (valores <= 0)).any(axis=1)
    invalidas |= (df["High"] < df["Low"]).to_numpy()
    if invalidas.any():
        fechas = ", ".join(df.index[invalidas][:5].strftime("%Y-%m-%d"))
        raise ValueError(f"Hay velas faltantes o inválidas ({fechas}). No se generó señal.")
    if len(df) < 200:
        raise ValueError("Se necesitan al menos 200 sesiones completas.")

    # Calcular la estrategia con OHLC ajustado. Mostrar el cierre sin ajuste
    # por dividendos y expresar el stop en la misma escala que ese cierre.
    factor = df["Adj Close"] / df["Close"]
    return pd.DataFrame({
        "High": df["High"] * factor,
        "Low": df["Low"] * factor,
        "Close": df["Adj Close"],
        "Cierre_USD": df["Close"],
        "Factor_Ajuste": factor,
    })


def wilder(serie: pd.Series, n: int = 14) -> pd.Series:
    """Inicializa con la media de los primeros n valores y suaviza con 1/n."""
    validos = serie.dropna()
    suavizada = pd.Series(np.nan, index=serie.index, dtype=float)
    if len(validos) < n:
        return suavizada
    tramo = validos.iloc[n - 1:].copy()
    tramo.iloc[0] = validos.iloc[:n].mean()
    suavizada.loc[tramo.index] = tramo.ewm(alpha=1 / n, adjust=False).mean()
    return suavizada


def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cierre = df["Close"]
    df["SMA_50"] = cierre.rolling(50).mean()
    df["SMA_200"] = cierre.rolling(200).mean()

    cambio = cierre.diff()
    subida = wilder(cambio.clip(lower=0))
    bajada = wilder((-cambio).clip(lower=0))
    total = subida + bajada
    # Solo subidas = 100, solo bajadas = 0, sin movimiento = 50.
    df["RSI_14"] = (100 * subida / total.where(total > 0)).mask(total == 0, 50.0)

    cierre_previo = cierre.shift()
    rango_real = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - cierre_previo).abs(),
        (df["Low"] - cierre_previo).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"] = wilder(rango_real)
    # SMA 200 es el indicador con el calentamiento más largo.
    return df.iloc[199:].copy()


def simular_senales(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye el estado teórico al cierre. No modela precios de ejecución."""
    dentro = False
    max_cierre = np.nan
    stop = np.nan
    filas = []

    for fila in df.itertuples():
        cierre = fila.Close
        distancia = ATR_MULT * fila.ATR_14
        nivel = np.nan
        motivo = ""

        if dentro:
            max_cierre = max(max_cierre, cierre)
            stop = max(stop, max_cierre - distancia)
            nivel = stop
            cruza_stop = cierre <= stop
            bajo_sma = cierre < fila.SMA_200
            if cruza_stop or bajo_sma:
                dentro = False
                max_cierre = stop = np.nan
                motivos = []
                if cruza_stop:
                    motivos.append("cierre en o por debajo del stop ATR")
                if bajo_sma:
                    motivos.append("cierre por debajo de la SMA 200")
                motivo = " y ".join(motivos)
                senal = "SALIDA TEÓRICA"
            else:
                senal = "MANTENER (posición teórica)"
        elif (
            cierre > fila.SMA_200
            and fila.SMA_50 > fila.SMA_200
            and fila.RSI_14 < RSI_ENTRADA
        ):
            dentro = True
            max_cierre = cierre
            stop = nivel = cierre - distancia
            senal = "ENTRADA TEÓRICA"
        else:
            senal = "SIN POSICIÓN (estado teórico)"

        filas.append((dentro, senal, nivel, motivo))

    resultado = pd.DataFrame(
        filas, index=df.index, columns=["Posicion", "Senal", "Nivel_Stop", "Motivo"]
    )
    return df.join(resultado)


def redactar_alerta(fila: pd.Series, sesion: pd.Timestamp, proxima: pd.Timestamp) -> str:
    factor = fila["Factor_Ajuste"]
    texto = [
        f"📊 {TICKER} | Cierre del {sesion.date()} (Nueva York)",
        f"Señal al cierre: {fila['Senal']}",
        f"Cierre Yahoo Finance: {fila['Cierre_USD']:.2f} USD",
        f"RSI (14): {fila['RSI_14']:.1f}",
        f"SMA 50: {fila['SMA_50'] / factor:.2f} USD",
        f"SMA 200: {fila['SMA_200'] / factor:.2f} USD",
        f"ATR (14): {fila['ATR_14'] / factor:.2f} USD",
    ]
    if pd.notna(fila["Nivel_Stop"]):
        texto.append(f"Umbral ATR al cierre: {fila['Nivel_Stop'] / factor:.2f} USD")
    if fila["Motivo"]:
        texto.append(f"Motivo: {fila['Motivo']}")
    if fila["Senal"] in ("ENTRADA TEÓRICA", "SALIDA TEÓRICA"):
        texto.append(f"Próxima apertura regular: {proxima:%Y-%m-%d %H:%M} (Nueva York).")
    texto.append("Modelo simulado. No consulta posiciones ni ejecuta órdenes.")
    return "\n".join(texto)


def enviar_una_vez(mensaje: str, sesion: pd.Timestamp) -> bool:
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise ValueError("Configura TELEGRAM_TOKEN y TELEGRAM_CHAT_ID en tu entorno.")

    ESTADO_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    fecha = sesion.date().isoformat()
    with closing(sqlite3.connect(ESTADO_LOCAL, timeout=30)) as db, db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS enviados (
                ticker TEXT, chat_id TEXT, fecha TEXT,
                PRIMARY KEY (ticker, chat_id, fecha)
            )
        """)
        db.execute("BEGIN IMMEDIATE")
        clave = (TICKER, chat_id, fecha)
        if db.execute(
            "SELECT 1 FROM enviados WHERE ticker=? AND chat_id=? AND fecha=?", clave
        ).fetchone():
            return False

        try:
            respuesta = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": mensaje},
                timeout=(5, 10),
                allow_redirects=False,
            )
        except requests.RequestException:
            # La URL contiene el token. No mostrar la excepción ni la URL.
            raise RuntimeError(
                "No se confirmó la respuesta de Telegram. Revisa el chat antes de reintentar."
            ) from None
        try:
            datos = respuesta.json()
        except ValueError:
            datos = {}
        if not 200 <= respuesta.status_code < 300 or not isinstance(datos, dict) or not datos.get("ok"):
            raise RuntimeError(f"Telegram no confirmó el envío (HTTP {respuesta.status_code}).")

        db.execute("INSERT INTO enviados VALUES (?, ?, ?)", clave)
    return True


def evaluar(enviar: bool = False, ahora: pd.Timestamp | None = None) -> str:
    ahora = pd.Timestamp.now(tz="UTC") if ahora is None else ahora
    calendario = calendario_de_sesiones(ahora)
    cerradas = calendario[calendario["market_close"] + ESPERA_PUBLICACION <= ahora]
    if cerradas.empty:
        raise ValueError("No hay una sesión cerrada en el calendario consultado.")

    ultima = cerradas.index[-1]
    hoy_ny = ahora.tz_convert(ZONA_MERCADO).date()
    if enviar and ultima.date() != hoy_ny:
        return "Sin envío: hoy no hay sesión o todavía no pasó el cierre más 25 minutos."

    historico = descargar_historico(cerradas.index)
    modelo = simular_senales(calcular_indicadores(historico))
    proxima = calendario.loc[calendario.index > ultima, "market_open"].iloc[0]
    mensaje = redactar_alerta(modelo.iloc[-1], ultima, proxima.tz_convert(ZONA_MERCADO))
    if not enviar:
        return mensaje
    if not enviar_una_vez(mensaje, ultima):
        return "Sin envío: ya se confirmó un reporte para este cierre."
    return f"{mensaje}\n\nEnvío confirmado por Telegram."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enviar", action="store_true", help="Envía el reporte al chat configurado.")
    args = parser.parse_args()
    try:
        print(evaluar(enviar=args.enviar))
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
        parser.exit(status=1, message=f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
