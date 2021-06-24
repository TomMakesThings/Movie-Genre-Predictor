from flaskr.main import app


def test_home_page_with_fixture(test_client):
    response = test_client.get('/')
    assert response.status_code == 200 # Successful connected


def test_home_page_post_with_fixture(test_client):
    """
    GIVEN a Flask application
    WHEN the '/' page is is posted to (POST)
    THEN check that a '400' status code is returned
    """
    response = test_client.post('/')
    assert response.status_code == 400


def test_result_page_post_with_fixture(test_client):

    response = test_client.post('/result')
    assert response.status_code == 500


def test_result_page_get_with_fixture(test_client):

    response = test_client.get('/result')
    assert response.status_code == 500