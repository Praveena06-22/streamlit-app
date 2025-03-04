from crop_recommendation_system.app import db, User

def delete_test_user():
    with db.session.begin():
        user = User.query.filter_by(username='test_user').first()
        if user:
            db.session.delete(user)
            db.session.commit()
            print("Test user deleted successfully!")
        else:
            print("No test user found.")

if __name__ == "__main__":
    delete_test_user()
