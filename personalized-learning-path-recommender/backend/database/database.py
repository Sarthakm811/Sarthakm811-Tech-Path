"""
Database Configuration and Management for Advanced Learning Path Recommender
Handles database connections, session management, and data operations
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import os
from typing import Generator
import logging

from .models import Base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database management class for handling connections and operations"""
    
    def __init__(self, database_url: str = None):
        """
        Initialize database manager
        
        Args:
            database_url: Database connection string. If None, uses environment variables
        """
        if database_url is None:
            # Default to PostgreSQL with environment variables
            db_host = os.getenv('DB_HOST', 'localhost')
            db_port = os.getenv('DB_PORT', '5432')
            db_name = os.getenv('DB_NAME', 'learning_recommender')
            db_user = os.getenv('DB_USER', 'postgres')
            db_password = os.getenv('DB_PASSWORD', 'password')
            
            database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        self.database_url = database_url
        
        # Create engine with connection pooling
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Database manager initialized with URL: {database_url}")
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions
        
        Usage:
            with db_manager.get_session() as session:
                # Use session here
                pass
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """
        Get a database session (manual management)
        
        Returns:
            SQLAlchemy session object
            
        Note: Remember to close the session when done!
        """
        return self.SessionLocal()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """Get database connection information"""
        try:
            with self.get_session() as session:
                # Get database version
                result = session.execute(text("SELECT version()"))
                db_version = result.scalar()
                
                # Get connection count
                result = session.execute(text("SELECT count(*) FROM pg_stat_activity"))
                connection_count = result.scalar()
                
                return {
                    'database_url': self.database_url,
                    'database_version': db_version,
                    'active_connections': connection_count,
                    'pool_size': self.engine.pool.size(),
                    'checked_in_connections': self.engine.pool.checkedin(),
                    'checked_out_connections': self.engine.pool.checkedout(),
                    'overflow_connections': self.engine.pool.overflow(),
                    'invalid_connections': self.engine.pool.invalid()
                }
        except Exception as e:
            logger.error(f"Error getting connection info: {e}")
            return {'error': str(e)}
    
    def execute_raw_sql(self, sql: str, params: dict = None):
        """Execute raw SQL query"""
        try:
            with self.get_session() as session:
                result = session.execute(text(sql), params or {})
                return result.fetchall()
        except Exception as e:
            logger.error(f"Error executing raw SQL: {e}")
            raise
    
    def backup_database(self, backup_path: str):
        """Create database backup (PostgreSQL specific)"""
        try:
            import subprocess
            
            # Extract database info from URL
            db_info = self._parse_database_url()
            
            # Create backup command
            cmd = [
                'pg_dump',
                '-h', db_info['host'],
                '-p', str(db_info['port']),
                '-U', db_info['user'],
                '-d', db_info['database'],
                '-f', backup_path,
                '--verbose'
            ]
            
            # Set password via environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = db_info['password']
            
            # Execute backup
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Database backup created successfully: {backup_path}")
                return True
            else:
                logger.error(f"Database backup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return False
    
    def _parse_database_url(self) -> dict:
        """Parse database URL into components"""
        from urllib.parse import urlparse
        
        parsed = urlparse(self.database_url)
        return {
            'host': parsed.hostname,
            'port': parsed.port,
            'user': parsed.username,
            'password': parsed.password,
            'database': parsed.path[1:]  # Remove leading slash
        }

# Global database manager instance
db_manager = None

def initialize_database(database_url: str = None) -> DatabaseManager:
    """
    Initialize the global database manager
    
    Args:
        database_url: Database connection string
        
    Returns:
        DatabaseManager instance
    """
    global db_manager
    db_manager = DatabaseManager(database_url)
    return db_manager

def get_database_manager() -> DatabaseManager:
    """
    Get the global database manager instance
    
    Returns:
        DatabaseManager instance
        
    Raises:
        RuntimeError: If database manager not initialized
    """
    if db_manager is None:
        raise RuntimeError("Database manager not initialized. Call initialize_database() first.")
    return db_manager

# Convenience functions for common operations
def get_session() -> Generator[Session, None, None]:
    """Get a database session using the global manager"""
    return get_database_manager().get_session()

def execute_query(sql: str, params: dict = None):
    """Execute a raw SQL query using the global manager"""
    return get_database_manager().execute_raw_sql(sql, params)

def test_database_connection() -> bool:
    """Test database connection using the global manager"""
    return get_database_manager().test_connection()

# Database initialization and setup
def setup_database(database_url: str = None, create_tables: bool = True) -> DatabaseManager:
    """
    Setup database with tables and initial data
    
    Args:
        database_url: Database connection string
        create_tables: Whether to create tables
        
    Returns:
        DatabaseManager instance
    """
    # Initialize database manager
    db_manager = initialize_database(database_url)
    
    # Test connection
    if not db_manager.test_connection():
        raise RuntimeError("Failed to connect to database")
    
    # Create tables if requested
    if create_tables:
        db_manager.create_tables()
    
    logger.info("Database setup completed successfully")
    return db_manager

# Example usage and testing
if __name__ == "__main__":
    # Example database setup
    try:
        # Setup database
        db = setup_database()
        
        # Test connection
        if test_database_connection():
            print("✅ Database connection successful")
        
        # Get connection info
        info = db.get_connection_info()
        print(f"📊 Database info: {info}")
        
        # Example query
        with db.get_session() as session:
            # Test query
            result = session.execute(text("SELECT current_timestamp"))
            timestamp = result.scalar()
            print(f"🕒 Current timestamp: {timestamp}")
            
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
