import random
import string

def get_random_id(n):
    characters = string.ascii_letters + string.digits  # letters and digits
    return ''.join(random.choice(characters) for _ in range(n))
