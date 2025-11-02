''' Run the Framework'''

docker-compose up --build --exit-code-from server | tee fl_logs/full_simulation.log