import asyncio


def proactor_loop_factory():
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    return asyncio.new_event_loop()
