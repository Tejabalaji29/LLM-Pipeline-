"""Centralized logger — import this everywhere."""
import sys
from loguru import logger
from configs.settings import LOGS_DIR

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}")
logger.add(LOGS_DIR / "pipeline_{time}.log", rotation="50 MB", retention="7 days", level="DEBUG")
