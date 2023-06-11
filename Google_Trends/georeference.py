"""
Given an address or a pair of GPS coordinates it returns with valuable geographical and topological information.
"""

import sqlalchemy as db
from geopy.geocoders import Nominatim

import auxiliary_functions as af


class Georeference:
    """Georeference class holds geographical and topological information for an address or a pair of coordinates.

    With an input of an address or a pair of coordinates for latitude and longitude, the address or the pair of
    coordinates gets analyzed and the retrieved results describe the location from the highest geographical and
    topological (e.g. country) level of information to the lowest (e.g. house number or type of open area (park,
    playground, etc.)).
    """

    def __init__(self, address=None, latitude=None, longitude=None, scheduler=False):
        """
        Parameters
        ----------
        address:
            str - the address of the location
        latitude:
            float - the geographical latitude coordinate of the location
        longitude:
            float - the geographical longitude coordinate of the location
        """
        if not scheduler:
            self._address = address
            self._latitude = latitude
            self._longitude = longitude
            self._scheduler = scheduler
            self._road, self._city, self._county, self._county_coords, self._state_district, self._state, self._country, \
                self._country_code = self._define_region()
            self._aux_county = self.__define_aux_county()
            self.flag = self._check_database()

    def _define_region(self):
        """Finds the city that the hotel is based on and the state's or larger area's information, e.g. county, state
        district and the geographical coordinates"""
        locator = Nominatim(user_agent='my_agent')
        if self._address:
            loc = locator.geocode(self._address, addressdetails=True, extratags=True, language='en', namedetails=True,
                                  timeout=100)
        else:
            coords = str(self._latitude) + ', ' + str(self._longitude)
            loc = locator.reverse(coords, language='en', addressdetails=True, timeout=100)
        self._data = loc.raw  # temp code

        road = loc.raw['address']['road']
        city = self._define_city(loc.raw['address'])
        county = self._define_county(loc.raw['address'])
        try:
            state, state_district = loc.raw['address']['state'], loc.raw['address']['state_district']
        except KeyError:
            state = state_district = None

        # get the general coordinates of the county
        loc = locator.geocode(city, addressdetails=True, extratags=True, language='en', namedetails=True, timeout=100)
        coordinates = [loc.latitude, loc.longitude]

        country = loc.raw['address']['country']
        country_code = loc.raw['address']['country_code'].upper()

        return road, city, county, coordinates, state_district, state, country, country_code

    @staticmethod
    def _define_county(address: dict) -> str:
        try:
            return address['county']
        except KeyError:
            try:
                return address['suburb']
            except KeyError:
                return Georeference._define_city(address)

    @staticmethod
    def _define_city(address: dict) -> str:
        try:
            return address['city']
        except KeyError:
            try:
                return address['town']
            except KeyError:
                try:
                    return address['village']
                except KeyError:
                    print('Something is wrong with the data')
                    return 'City'

    @staticmethod
    def _define_street(address):
        pass  # try:

    def __define_aux_county(self):
        parts = self._county.split()
        if self._city in parts:
            return self._city
        else:
            return self._county

    def _check_database(self):
        """Checks whether or not the data exist in the database.

        Returns
        -------
        boolean:
             flag - indicates whether or not the topological information is already stored in the database
        """
        engine, connection, metadata = af.create_connection()

        select = db.text("SELECT * FROM aihg_counties WHERE name = :x")
        results = list(connection.execute(select, x=self._county))
        if results:
            flag = False
        else:
            self._insert_county(connection, metadata)
            self._insert_city(connection, metadata)
            flag = True
        af.close_connection(engine, connection)
        return flag

    def _insert_county(self, connection, metadata):
        """Inserts a county into the database"""
        select = db.text("SELECT code FROM aihg_countries WHERE name = :x")
        results = list(connection.execute(select, x=self._country))

        insert = db.insert(metadata.tables['aihg_counties']).values(
            name=self._county, aux_term=self._aux_county, country_code=results[0][0],
            latitude=self._county_coords[0], longitude=self._county_coords[1]
        )
        _ = connection.execute(insert)

    def _insert_city(self, connection, metadata):
        """Inserts a city into the database and gets updated with a county id"""
        select = db.text("SELECT ID, country_code FROM aihg_counties WHERE name = :x")
        results = connection.execute(select, x=self._county)
        county_id, country_code = results.fetchall()[0]

        insert = db.insert(metadata.tables['aihg_cities']).values(name=self._city, county_ID=county_id,
                                                                  country_code=country_code)
        _ = connection.execute(insert)
