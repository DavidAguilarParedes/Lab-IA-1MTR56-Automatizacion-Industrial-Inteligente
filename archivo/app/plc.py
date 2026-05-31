"""
Puente de comunicación con PLC Beckhoff via pyads (TwinCAT 3).

Los alumnos completan las funciones marcadas con TODO.
El código comentado muestra exactamente qué hacer — solo descomentan y ajustan.
"""

import time
import logging

logger = logging.getLogger(__name__)

try:
    import pyads
    PYADS_AVAILABLE = True
except ImportError:
    PYADS_AVAILABLE = False


class PLCBridge:
    """Puente de comunicación con PLC Beckhoff via ADS/pyads."""

    def __init__(self):
        self.plc = None
        self.connected = False
        self.ams_net_id = ""
        self.port = 851
        self._log = []
        self._prev_inicio = False  # para detección de flanco

    def connect(self, ams_net_id, port=851):
        """Conectar al PLC Beckhoff via ADS.

        Args:
            ams_net_id: AMS Net ID del PLC (ej: '5.80.201.232.1.1')
            port: Puerto ADS (851 para TwinCAT 3 Runtime 1)

        Returns:
            (bool, str): (éxito, mensaje)
        """
        if not PYADS_AVAILABLE:
            return False, "Error: librería 'pyads' no instalada. Ejecute: pip install pyads"

        try:
            # TODO ALUMNO: Completar conexión pyads
            # Descomentar las siguientes líneas:
            # self.plc = pyads.Connection(ams_net_id, port)
            # self.plc.open()
            # self.connected = True

            self.ams_net_id = ams_net_id
            self.port = port

            # --- Placeholder mientras no se completa ---
            self.connected = False
            msg = f"TODO: Completar connect() en app/plc.py"
            self._add_log(msg, "⚠")
            return False, msg

            # Cuando esté completado, cambiar a:
            # msg = f"Conectado a {ams_net_id}:{port}"
            # self._add_log(msg, "✓")
            # return True, msg

        except Exception as e:
            self.connected = False
            msg = f"Error de conexión: {e}"
            self._add_log(msg, "✗")
            return False, msg

    def disconnect(self):
        """Desconectar del PLC."""
        if self.plc and self.connected:
            try:
                # TODO ALUMNO: Cerrar conexión
                # self.plc.close()
                pass
            except Exception:
                pass
        self.connected = False
        self.plc = None
        self._add_log("Desconectado", "—")

    def leer_inicio(self, variable='GVL.bInicioControlDeCalidad'):
        """Leer señal de inicio de inspección del PLC.

        Args:
            variable: Nombre de la variable BOOL en el PLC

        Returns:
            bool: True si el PLC pide iniciar inspección
        """
        if not self.connected:
            return False

        try:
            # TODO ALUMNO: Leer variable BOOL del PLC
            # Descomentar la siguiente línea:
            # return self.plc.read_by_name(variable, pyads.PLCTYPE_BOOL)

            return False  # placeholder

        except Exception as e:
            logger.warning(f"Error leyendo {variable}: {e}")
            return False

    def detectar_flanco(self, variable='GVL.bInicioControlDeCalidad'):
        """Detecta flanco de subida (False→True) en la señal de inicio.

        Returns:
            bool: True solo en el momento de la transición False→True
        """
        actual = self.leer_inicio(variable)
        flanco = actual and not self._prev_inicio
        self._prev_inicio = actual
        return flanco

    def enviar_resultado(self, clase_int, confianza,
                         var_clase='GVL.nResultadoClase',
                         var_confianza='GVL.rConfianza'):
        """Escribir resultado de clasificación al PLC.

        Args:
            clase_int: Índice de la clase detectada (INT)
            confianza: Nivel de confianza 0.0-1.0 (REAL)
            var_clase: Variable PLC para la clase (INT)
            var_confianza: Variable PLC para la confianza (REAL)

        Returns:
            (bool, str): (éxito, mensaje)
        """
        if not self.connected:
            return False, "No conectado al PLC"

        try:
            # TODO ALUMNO: Escribir resultado al PLC
            # Descomentar las siguientes líneas:
            # self.plc.write_by_name(var_clase, clase_int, pyads.PLCTYPE_INT)
            # self.plc.write_by_name(var_confianza, confianza, pyads.PLCTYPE_REAL)

            # Placeholder:
            msg = f"TODO: Completar enviar_resultado() — clase={clase_int}, conf={confianza:.1%}"
            self._add_log(msg, "⚠")
            return False, msg

            # Cuando esté completado, cambiar a:
            # msg = f"Enviado: clase={clase_int}, conf={confianza:.1%}"
            # self._add_log(msg, "✓")
            # return True, msg

        except Exception as e:
            msg = f"Error al escribir: {e}"
            self._add_log(msg, "✗")
            return False, msg

    def _add_log(self, msg, icon=""):
        ts = time.strftime("%H:%M:%S")
        entry = f"{ts}  {icon} {msg}"
        self._log.append(entry)
        logger.info(entry)

    def get_log(self, last_n=50):
        """Retorna las últimas N líneas del log."""
        return "\n".join(self._log[-last_n:])

    @property
    def status_emoji(self):
        return "🟢 Conectado" if self.connected else "🔴 Desconectado"

    @property
    def status_text(self):
        if self.connected:
            return f"Conectado — {self.ams_net_id}:{self.port}"
        return "Desconectado"


# Instancia global
plc_bridge = PLCBridge()
