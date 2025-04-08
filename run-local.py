import multiprocessing
import os
from rq import Worker, Queue
from redis import Redis
from dotenv import load_dotenv

# Load local environment variables
load_dotenv('.env.local')


def run_server():
    from biocentral_server.server_entrypoint import AppState
    print("Starting Biocentral Server")
    app_state = AppState.get_instance()
    app = app_state.init_app()
    app_state.init_app_context()
    app.run(debug=True, port=9540, use_reloader=False)


def run_worker(worker_id):
    """Run a single worker process with a specific name"""
    redis_jobs_host = os.environ.get("REDIS_JOBS_HOST")
    redis_jobs_port = os.environ.get("REDIS_JOBS_PORT")
    redis_conn = Redis(host=redis_jobs_host, port=redis_jobs_port)

    # Use custom worker name to identify in monitoring
    worker_name = f"biocentral-worker-{worker_id}"

    # Configure worker with appropriate timeouts for long-running tasks
    worker = Worker(
        queues=['high', 'default', 'low'],  # Process high priority queue first
        connection=redis_conn,
        name=worker_name,
        worker_ttl=600,  # 10 minutes heartbeat
        default_result_ttl=600,  # Keep results for 10 minutes
        job_monitoring_interval=2,  # Check job status every 2 seconds
    )

    print(f"Starting worker {worker_name}")
    worker.teardown()
    worker.work()


def start_workers(num_workers=4):
    """Start multiple worker processes"""
    worker_processes = []
    for i in range(num_workers):
        process = multiprocessing.Process(
            target=run_worker,
            args=(i,),
            name=f"rq-worker-process-{i}"
        )
        process.start()
        worker_processes.append(process)
    return worker_processes


if __name__ == '__main__':
    # Determine number of workers based on CPU count
    cpu_count = multiprocessing.cpu_count()

    num_workers = 4

    print(f"Starting {num_workers} RQ worker processes")
    worker_processes = start_workers(num_workers)

    try:
        # Run the server in the main process
        run_server()
    finally:
        # Ensure all workers are terminated when server stops
        print("Shutting down worker processes...")
        for process in worker_processes:
            process.terminate()
            process.join(timeout=2)  # Wait up to 2 seconds for each worker
            if process.is_alive():
                print(f"Force terminating worker {process.name}")
                os.kill(process.pid, 9)  # SIGKILL if still alive
