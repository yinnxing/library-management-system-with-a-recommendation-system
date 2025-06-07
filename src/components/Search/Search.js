// Search.js
import React from 'react';
import SearchItem from '../SearchItem/searchItem';
import styles from './Search.module.css';
import UserApi from '../../api/UserApi';
import { useState, useEffect, useRef } from 'react';

const Search = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [results, setResults] = useState([]);
    const [allBooks, setAllBooks] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const searchInputRef = useRef(null);

    // Fetch all books when component mounts
    useEffect(() => {
        const fetchAllBooks = async () => {
            try {
                setLoading(true);
                const response = await UserApi.getBooks(1, 1000); // Get a large number to fetch all books
                if (response.data.code === 0) {
                    setAllBooks(response.data.result.content);
                }
            } catch (error) {
                console.error('Error fetching books:', error);
                setError('Failed to load books');
            } finally {
                setLoading(false);
            }
        };

        fetchAllBooks();
    }, []);

    const handleSearch = (e) => {
        const value = e.target.value;
        setSearchTerm(value);

        if (value.trim() === '') {
            setResults([]);
            return;
        }

        // Filter books based on title, author, or genre
        const filteredBooks = allBooks.filter((book) =>
            book.title.toLowerCase().includes(value.toLowerCase()) ||
            book.author.toLowerCase().includes(value.toLowerCase()) ||
            (book.genre && book.genre.toLowerCase().includes(value.toLowerCase()))
        );
        
        // Limit results to prevent too many items
        setResults(filteredBooks.slice(0, 10));
    };

    const handleClickOutside = (e) => {
        if (searchInputRef.current && !searchInputRef.current.contains(e.target)) {
            setResults([]);  
        }
    };

    useEffect(() => {
        document.addEventListener('mousedown', handleClickOutside);
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, []);

    const handleSelect = (title) => {
        console.log('Selected:', title);
        setSearchTerm(title);
        setResults([]); // Close results after selection
    };

    return (
        <div className={styles.search} ref={searchInputRef}>
            <input
                type="text"
                value={searchTerm}
                onChange={handleSearch}
                placeholder="Tìm kiếm sách theo tên, tác giả, thể loại..."
                disabled={loading}
            />
            {loading && (
                <div className={styles.loadingContainer}>
                    <div className={styles.loadingSpinner}></div>
                    <span>Đang tải...</span>
                </div>
            )}
            {error && (
                <div className={styles.errorContainer}>
                    <span className={styles.errorIcon}>⚠️</span>
                    <span>{error}</span>
                </div>
            )}
            {results.length > 0 && !loading && (
                <div className={styles.resultsContainer}>
                    {results.map((result) => (
                        <SearchItem
                            key={result.bookId} 
                            book={result}  
                            onSelect={handleSelect}  
                        />
                    ))}
                    {results.length === 10 && (
                        <div className={styles.moreResults}>
                            <span>Và nhiều kết quả khác...</span>
                        </div>
                    )}
                </div>
            )}
            {searchTerm.trim() !== '' && results.length === 0 && !loading && !error && (
                <div className={styles.noResults}>
                    <span className={styles.noResultsIcon}>📚</span>
                    <span>Không tìm thấy sách nào phù hợp</span>
                </div>
            )}
        </div>
    );
};

export default Search;
