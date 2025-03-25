import mysql.connector
import bcrypt

from db_config import DB_CONFIG
import mysql.connector

def connect_to_db():
    return mysql.connector.connect(**DB_CONFIG)

try:
    conn = connect_to_db()
    print("Połączono z bazą danych!")
    conn.close()
except mysql.connector.Error as err:
    print(f"Błąd: {err}")

def verify_login(email, password):
    connection = connect_to_db()
    cursor = connection.cursor()
    query = "SELECT id,password, admin FROM user WHERE email = %s"
    cursor.execute(query, (email,))
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    if result:
        user_id, hashed_password, admin = result
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            return admin,user_id
    return None, None

def create_user(email, password, position_id, is_admin, name, surname):
    connection = connect_to_db()
    cursor = connection.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        query = "INSERT INTO user (name,surname,email, password, position_id, admin) VALUES (%s, %s, %s, %s, %s, %s)"
        cursor.execute(query, (name,surname,email, hashed_password, position_id, is_admin))
        connection.commit()
        return True
    except Exception as e:
        print("Błąd podczas dodawania użytkownika:", e)
        connection.rollback()
        return False
    finally:
        cursor.close()
        connection.close()

def get_user_data(user_id):
    connection = connect_to_db()
    cursor = connection.cursor(dictionary=True)
    query = "SELECT u.name, u.surname,u.email, p.name AS position_name FROM user u LEFT JOIN `position` p ON u.position_id = p.id WHERE u.id = %s"
    cursor.execute(query, (user_id,))
    user_data = cursor.fetchone()
    cursor.close()
    connection.close()
    return user_data

def get_positions():
    connection = connect_to_db()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM `position`")
    positions = cursor.fetchall()
    cursor.close()
    connection.close()
    return positions



def update_email(user_id, new_email):
    connection = connect_to_db()
    cursor = connection.cursor()
    try:
        query = "UPDATE user SET email = %s WHERE id = %s"
        cursor.execute(query, (new_email, user_id))
        connection.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print("Błąd podczas aktualizacji emaila:", e)
        connection.rollback()
        return False
    finally:
        cursor.close()
        connection.close()

def update_password(user_id, current_password, new_password):
    connection = connect_to_db()
    cursor = connection.cursor()
    try:
        # Pobranie aktualnego hasła
        cursor.execute("SELECT password FROM user WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        if not result:
            return False
        hashed_password = result[0]
        if not bcrypt.checkpw(current_password.encode('utf-8'), hashed_password.encode('utf-8')):
            return False
        # Zmiana hasła
        new_hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute("UPDATE user SET password = %s WHERE id = %s", (new_hashed_password, user_id))
        connection.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print("Błąd podczas aktualizacji hasła:", e)
        connection.rollback()
        return False
    finally:
        cursor.close()
        connection.close()

def get_users():
    connection = connect_to_db()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT u.id, u.name, u.surname, u.email,u.admin, p.name AS position_name FROM user u LEFT JOIN `position` p ON u.position_id = p.id")
    users = cursor.fetchall()
    cursor.close()
    connection.close()
    return users

def delete_user(user_id):
    connection = connect_to_db()
    cursor = connection.cursor()
    try:
        cursor.execute("DELETE FROM user WHERE id = %s", (user_id,))
        connection.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print("Błąd podczas usuwania użytkownika:", e)
        connection.rollback()
        return False
    finally:
        cursor.close()
        connection.close()

def edit_user(user_id, name, surname, email, position_id, is_admin):
    connection = connect_to_db()
    cursor = connection.cursor()
    try:
        cursor.execute("UPDATE user SET name = %s, surname = %s, email = %s, position_id = %s, admin = %s WHERE id = %s",
                       (name, surname, email, position_id, is_admin, user_id))
        connection.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print("Błąd podczas aktualizacji użytkownika:", e)
        connection.rollback()
        return False
    finally:
        cursor.close()
        connection.close()
