from biocentral_server.utils import Constants
from biocentral_server.server_entrypoint import AppState

# Create and initialize the application
app_state = AppState.get_instance()
app = app_state.init_app()
app_state.init_app_context()

if __name__ == '__main__':
    # For development environments
    app.run(port=Constants.SERVER_DEFAULT_PORT)
