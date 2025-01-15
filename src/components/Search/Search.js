// Search.js
import React from 'react';
import SearchItem from '../SearchItem/searchItem';
import styles from './Search.module.css';
import books from '../../assets/all_books'
import { useState, useEffect, useRef } from 'react';
const Search = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [results, setResults] = useState([]);
    const searchInputRef = useRef(null);

    const handleSearch = (e) => {
        const value = e.target.value;
        setSearchTerm(value);

        if (value.trim() === '') {
            setResults([]);
            return;
        }

        const filteredBooks = books.filter((book) =>
            book.title.toLowerCase().includes(value.toLowerCase())
        );
        setResults(filteredBooks);
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
        setSearchTerm(title);  // Chỉ thay đổi giá trị searchTerm
        // Không cần gọi setResults([]) ở đây vì không muốn đóng kết quả tìm kiếm
    };

    return (
        <div className={styles.search} ref={searchInputRef}>
            <input
                type="text"
                value={searchTerm}
                onChange={handleSearch}
                placeholder="Tìm kiếm sách..."
            />
            {results.length > 0 && (
                <div className={styles.resultsContainer}>
                    {results.map((result) => (
                        <SearchItem
                            key={result.bookId} 
                            book={result}  
                            onSelect={handleSelect}  
                        />
                    ))}
                </div>
            )}
        </div>
    );
};



export default Search;
