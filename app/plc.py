"""Cliente OPC UA para comunicación con PLC."""

import time
import logging

logger = logging.getLogger(__name__)

try:
    from opcua import Client, ua
    OPCUA_AVAILABLE = True
except ImportError:
    OPCUA_AVAILABLE = False


class PLCBridge:
    """Puente de comunicación con PLC via OPC UA."""

    def __init__(self):
        self.client = None
        self.connected = False
        self.url = ""
        self._log = []

    def connect(self, url):
        """Conectar al servidor OPC UA."""
        if not OPCUA_AVAILABLE:
            return False, "Error: librería 'opcua' no instalada. Ejecute: uv sync"
        try:
            self.client = Client(url)
            self.client.connect()
            self.connected = True
            self.url = url
            msg = f"Conectado a {url}"
            self._add_log(msg, "✓")
            return True, msg
        except Exception as e:
            self.connected = False
            msg = f"Error de conexión: {e}"
            self._add_log(msg, "✗")
            return False, msg

    def disconnect(self):
        """Desconectar del servidor."""
        if self.client and self.connected:
            try:
                self.client.disconnect()
            except Exception:
                pass
        self.connected = False
        self.client = None
        self._add_log("Desconectado", "—")

    def send_classification(self, node_id, value, conf_node_id=None,
                            confidence=None):
        """Escribe clasificación en nodo OPC UA.

        Args:
            node_id: nodo OPC UA (ej: "ns=2;s=Clasificacion").
            value: valor entero a escribir.
            conf_node_id: nodo para confianza (opcional).
            confidence: valor float de confianza (opcional).
        """
        if not self.connected:
            return False, "No conectado al PLC"
        try:
            node = self.client.get_node(node_id)
            node.set_value(ua.Variant(int(value), ua.VariantType.Int16))

            if conf_node_id and confidence is not None:
                conf_node = self.client.get_node(conf_node_id)
                conf_node.set_value(
                    ua.Variant(float(confidence), ua.VariantType.Float)
                )

            conf_str = f" (conf={confidence:.1%})" if confidence is not None else ""
            msg = f"valor={value}{conf_str}"
            self._add_log(msg, "✓")
            return True, msg
        except Exception as e:
            msg = f"Error al escribir: {e}"
            self._add_log(msg, "✗")
            return False, msg

    def _add_log(self, msg, icon=""):
        ts = time.strftime("%H:%M:%S")
        self._log.append(f"{ts}  {icon} {msg}")

    def get_log(self, last_n=50):
        """Retorna las últimas N líneas del log."""
        return "\n".join(self._log[-last_n:])

    @property
    def status_emoji(self):
        return "🟢 Conectado" if self.connected else "🔴 Desconectado"


# Instancia global
plc_bridge = PLCBridge()
